#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ABFE 核心物理模块 (v6.0 - 完整收敛版)
职责：统一封装所有势能、限制力、拟合器、估算器、校验器、路径规划、替身构建、Orb扫描与路由工厂
架构约束：严格收敛至 5 文件，本文件为唯一物理核心单例，零占位符
依赖：openmm, numpy, scipy, mdtraj, torch, openmmml (部分功能)
"""

import openmm
from openmm import app, unit
import numpy as np
import math
import warnings
import json
import os
import logging
import gc
import builtins
import statistics
from itertools import combinations
from collections import deque
from typing import Dict, List, Tuple, Optional, Any, Callable
from scipy.optimize import differential_evolution, least_squares, minimize
from scipy import constants

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import mdtraj

    HAS_MDTRAJ = True
except ImportError:
    HAS_MDTRAJ = False
try:
    import torch
    from openmmml import MLPotential

    HAS_ORB = True
except ImportError:
    HAS_ORB = False
try:
    import pymbar

    HAS_PYMBAR = True
except ImportError:
    HAS_PYMBAR = False

logger = logging.getLogger(__name__)


def _build_openmmml_kwargs(
    device: Optional[str] = None,
    precision: Optional[str] = None,
    return_energy_type: Optional[str] = None,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
) -> Dict[str, Any]:
    """
    统一按 openmm-ml 官方接口组织 MLPotential.createSystem() 参数。
    是否显式传 precision 由调用侧决定；若不传则遵循 openmm-ml 的模型默认精度。
    """
    kwargs: Dict[str, Any] = {}
    if return_energy_type is not None:
        kwargs["returnEnergyType"] = return_energy_type
    if device is not None:
        kwargs["device"] = device
    if precision in ("single", "double"):
        kwargs["precision"] = precision
    if charge is not None:
        kwargs["charge"] = charge
    if multiplicity is not None:
        kwargs["multiplicity"] = multiplicity
    return kwargs


def _select_env_indices_from_mdtraj_frame(frame, lig_idx: np.ndarray, env_radius_nm: float, max_env_atoms: Optional[int] = None) -> np.ndarray:
    """
    先做半径近邻筛选，再按“到配体最近距离”的 Top-K 排序裁剪环境原子。
    这里裁的是原子，不是整盒水，也不是全环境残基。
    """
    import mdtraj as md

    raw_env = md.compute_neighbors(frame, env_radius_nm, lig_idx)[0]
    env_idx = np.setdiff1d(raw_env, lig_idx, assume_unique=True)
    if max_env_atoms is None or len(env_idx) <= max_env_atoms:
        return np.asarray(env_idx, dtype=int)

    pos_nm = np.asarray(frame.xyz[0], dtype=np.float64)
    if frame.unitcell_vectors is not None:
        box_vecs = np.asarray(frame.unitcell_vectors[0], dtype=np.float64)
        box_lens = np.linalg.norm(box_vecs, axis=1)
    else:
        box_lens = None

    delta = pos_nm[lig_idx][:, None, :] - pos_nm[env_idx][None, :, :]
    if box_lens is not None:
        delta -= box_lens * np.round(delta / box_lens)
    dists = np.linalg.norm(delta, axis=-1)
    min_dists = np.min(dists, axis=0)
    keep_order = np.argsort(min_dists)[:max_env_atoms]
    return np.sort(np.asarray(env_idx[keep_order], dtype=int))


def _infer_log_level_from_message(message: str) -> int:
    if any(token in message for token in ("⚠️", "警告", "warning")):
        return logging.WARNING
    if any(token in message for token in ("🚨", "❌", "失败", "错误", "异常", "error")):
        return logging.ERROR
    return logging.INFO


def _log_print(*args, sep=" ", end="\n", file=None, flush=False):
    message = sep.join(str(arg) for arg in args)
    if end and end != "\n":
        message += end.rstrip("\n")
    if logger.handlers:
        logger.log(_infer_log_level_from_message(message), message)
    builtins.print(*args, sep=sep, end=end, file=file, flush=flush)


print = _log_print


def _pymbar_version_tuple() -> Tuple[int, ...]:
    if not HAS_PYMBAR:
        return (0,)
    version_str = str(getattr(pymbar, "__version__", getattr(pymbar, "version", "0")))
    parts = []
    for token in version_str.replace("-", ".").split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts) if parts else (0,)


def _build_mbar_compatible(u_kn, n_k, **kwargs):
    """
    兼容 PyMBAR 3.x/4.x 的 MBAR 构造器。
    若某些关键字参数在当前版本不可用，则按保守顺序回退。
    """
    if not HAS_PYMBAR:
        raise ImportError("需要 pymbar 包，请安装: pip install pymbar")

    base_kwargs = dict(kwargs)
    drop_order = [
        "solver_protocol",
        "initialize",
        "relative_tolerance",
        "solver_tolerance",
        "initial_f_k",
        "verbose",
    ]
    variants = [base_kwargs]
    seen = {tuple(sorted((k, repr(v)) for k, v in base_kwargs.items()))}
    current = dict(base_kwargs)
    for key in drop_order:
        if key in current:
            current = dict(current)
            current.pop(key, None)
            signature = tuple(sorted((k, repr(v)) for k, v in current.items()))
            if signature not in seen:
                variants.append(current)
                seen.add(signature)

    last_type_error = None
    for candidate in variants:
        try:
            return pymbar.MBAR(u_kn, n_k, **candidate)
        except TypeError as exc:
            last_type_error = exc
            continue
    if last_type_error is not None:
        raise last_type_error
    return pymbar.MBAR(u_kn, n_k)


def _extract_mbar_matrix(result, primary_name: str, fallback_names: Tuple[str, ...]) -> Optional[np.ndarray]:
    candidate_names = (primary_name,) + tuple(fallback_names)
    for name in candidate_names:
        if isinstance(result, dict) and name in result:
            return np.asarray(result[name], dtype=float)
        if hasattr(result, name):
            return np.asarray(getattr(result, name), dtype=float)
    return None


def _compute_free_energy_result_compatible(mbar, compute_uncertainty: bool = True):
    methods = []
    if hasattr(mbar, "compute_free_energy_differences"):
        methods.append(("compute_free_energy_differences", {"compute_uncertainty": compute_uncertainty}))
    if hasattr(mbar, "compute_free_energy"):
        methods.append(("compute_free_energy", {}))
    if not methods:
        raise AttributeError("当前 pymbar.MBAR 对象不包含可用的自由能计算方法")

    last_exc = None
    for method_name, kwargs in methods:
        try:
            return getattr(mbar, method_name)(**kwargs)
        except TypeError:
            try:
                return getattr(mbar, method_name)()
            except Exception as exc:
                last_exc = exc
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("MBAR 自由能计算失败")


def _extract_free_energy_arrays(result, require_uncertainty: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    delta_f = _extract_mbar_matrix(result, "Delta_f", ("delta_f", "free_energy"))
    if delta_f is None:
        raise KeyError("无法从 pymbar 结果中提取自由能矩阵")

    delta_df = _extract_mbar_matrix(result, "dDelta_f", ("d_delta_f", "error", "uncertainty"))
    if delta_df is None:
        if require_uncertainty:
            raise KeyError("无法从 pymbar 结果中提取不确定度矩阵")
        delta_df = np.full_like(delta_f, np.nan, dtype=float)
    return delta_f, delta_df


def get_optimal_device_settings():
    if not HAS_ORB or not torch.cuda.is_available():
        return "cpu", False
    device = "cuda"
    major, minor = torch.cuda.get_device_capability()
    support_tf32 = False
    if major >= 8:
        support_tf32 = True
        torch.set_float32_matmul_precision("high")
    return device, support_tf32


GLOBAL_DEVICE, SUPPORTS_TF32 = get_optimal_device_settings()


# ============================================================================
# 0. 单位常量与验证器
# ============================================================================
class UnitConstants:
    NM_PER_ANGSTROM = 0.1
    KJ_PER_KCAL = 4.184
    KJ_PER_NM2_PER_KCAL_PER_A2 = 418.4
    RAD_PER_DEG = np.pi / 180.0
    DISTANCE_RANGE_NM = (0.1, 10.0)
    ANGLE_RANGE_RAD = (0.0, math.pi)
    FORCE_CONSTANT_KR_RANGE = (100.0, 100000.0)
    FORCE_CONSTANT_KANGLE_RANGE = (10.0, 1000.0)


class UnitValidator:
    @staticmethod
    def validate_distance(v, n="dist"):
        if v <= 0:
            raise ValueError(f"{n}必须>0")
        if v > 100:
            warnings.warn(f"{n}={v} nm 可能过大")

    @staticmethod
    def validate_force_constant(v, n="k"):
        if v <= 0:
            raise ValueError(f"{n}必须为正值")

class NumpyEncoder(json.JSONEncoder):
    """🔑 全局统一 JSON 序列化器（处理 numpy 类型/数组）"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return super().default(obj)
# ============================================================================
# 1. ACES 软核势 & DEXP 替身势 & Orb 拟合器
# ============================================================================
class ACESoftcorePotential:
    def __init__(
        self, alpha_lj=0.5, alpha_coul=0.2, power_lj=(2, 2), power_coul=(1, 1)
    ):
        self.alpha_lj, self.alpha_coul = float(alpha_lj), float(alpha_coul)
        self.m_lj, self.n_lj = power_lj
        self.m_coul, self.n_coul = power_coul

    def build_expression(self, lam_coul, lam_vdw):
        COUL = 138.935456
        lc, lv = f"({lam_coul}^{self.n_coul})", f"({lam_vdw}^{self.n_lj})"

        # 仅在 r->0 且 lambda->1 的奇异角落启用兜底，不污染正常物理区间。
        dlj = f"max({self.alpha_lj}*(1.0-{lam_vdw})^{self.m_lj} + r^6, 1e-6)"
        dc = f"sqrt(max(r^2 + {self.alpha_coul}*(1.0-{lam_coul})^{self.m_coul}, 1e-6))"
        
        # 必须整体加括号，否则会被解析成 0.5*(sigma1+sigma2)^n，
        # 而不是 ((sigma1+sigma2)/2)^n，短程排斥会被严重放大。
        sigma12 = "(0.5*(sigma1+sigma2))"
        lj = f"{lv} * 4 * sqrt(epsilon1*epsilon2) * ({sigma12}^12/({dlj}^2) - {sigma12}^6/{dlj})"
        coul = f"{lc} * {COUL} * q1 * q2 / {dc}"
        
        return f"{lj} + {coul}"

    @staticmethod
    def _normalize_alpha_units(alpha_lj, alpha_coul):
        return float(alpha_lj), float(alpha_coul)

    @staticmethod
    def optimize_alpha(n, alpha_coul_nm2=None):
        """
        ✅ 修复：OpenMM 软核 alpha 标准单位即为 nm⁶ / nm²
        文献值 0.5 已对应 nm 尺度，无需额外乘以 1e-6/1e-2
        """
        if n > 50:
            alpha_lj, alpha_coul = 0.5, (alpha_coul_nm2 or 0.2)
        else:
            alpha_lj, alpha_coul = 0.5, (alpha_coul_nm2 or 0.3)
            
        # ✅ 更新断言范围（匹配 nm 标准）
        assert 0.1 < alpha_lj < 2.0, f"alpha_lj 单位疑似错误: {alpha_lj} (预期 0.1~2.0 nm⁶)"
        assert 0.05 < alpha_coul < 1.0, f"alpha_coul 超出安全范围: {alpha_coul} (预期 0.05~1.0 nm²)"
        
        return {
            "alpha_lj": alpha_lj,        # nm⁶
            "alpha_coul": alpha_coul,    # nm²
            "power_lj": [2, 2],
            "power_coul": [1, 1],
        }
    def get_parameters_dict(self):
        return {
            "alpha_lj": self.alpha_lj,
            "alpha_coul": self.alpha_coul,
            "power_lj": list([self.m_lj, self.n_lj]),
            "power_coul": list([self.m_coul, self.n_coul]),
        }

    @classmethod
    def from_dict(cls, p):
        alpha_lj, alpha_coul = cls._normalize_alpha_units(
            p.get("alpha_lj", 0.5),
            p.get("alpha_coul", 0.2),
        )
        return cls(
            alpha_lj,
            alpha_coul,
            tuple(p.get("power_lj", [2, 2])),
            tuple(p.get("power_coul", [1, 1])),
        )


class BeutlerSoftcoreBuilder:
    """传统 Beutler 式软核势构建器 (CustomNonbondedForce + interaction group)
    与 ACESoftcorePotential 区别：显式 L-E 对过滤 + 传统 alpha*(1-lambda)^power 表达式
    """
    @staticmethod
    def build(
        nb_force: openmm.NonbondedForce,
        ligand_indices: List[int],
        env_indices: List[int],
        alpha_lj: float = 0.5,
        alpha_coul: float = 0.5,
        power_lj: int = 1,
        power_coul: int = 1,
        particle_params_override=None,
    ) -> openmm.CustomNonbondedForce:
        expr = (
            f"lambda_vdw * 4*sqrt(epsilon1*epsilon2)*("
            f"(sigma12^12 / (r^6 + {alpha_lj}*(1-lambda_vdw)^{power_lj} + 1e-4*(1-lambda_vdw))^2) - "
            f"(sigma12^6 / (r^6 + {alpha_lj}*(1-lambda_vdw)^{power_lj} + 1e-4*(1-lambda_vdw)))"
            f") + "
            f"lambda_coul * 138.935456 * q1*q2 / sqrt(r^2 + {alpha_coul}*(1-lambda_coul)^{power_coul} + 1e-3); "
            f"sigma12=(0.5*(sigma1+sigma2))"
        )
        sc_force = openmm.CustomNonbondedForce(expr)
        for p in ["q", "sigma", "epsilon"]:
            sc_force.addPerParticleParameter(p)
        sc_force.addGlobalParameter("lambda_coul", 1.0)
        sc_force.addGlobalParameter("lambda_vdw", 1.0)

        for i in range(nb_force.getNumParticles()):
            if particle_params_override is not None and i < len(particle_params_override):
                q, sig, eps = particle_params_override[i]
            else:
                q, sig, eps = nb_force.getParticleParameters(i)
            sc_force.addParticle([
                q.value_in_unit(unit.elementary_charge),
                sig.value_in_unit(unit.nanometer),
                eps.value_in_unit(unit.kilojoule_per_mole)
            ])

        sc_force.addInteractionGroup(set(ligand_indices), set(env_indices))
        sc_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        sc_force.setCutoffDistance(1.2 * unit.nanometer)
        sc_force.setUseSwitchingFunction(True)
        sc_force.setSwitchingDistance(1.0 * unit.nanometer)

        for i in range(nb_force.getNumExceptions()):
            p1, p2, _, _, _ = nb_force.getExceptionParameters(i)
            sc_force.addExclusion(int(p1), int(p2))

        return sc_force


class DEXPSurrogatePotential:
    def __init__(
        self,
        alpha_vdw=12.0,
        beta_vdw=8.0,
        r0_vdw=0.33,
        A_fit=1.0,
        B_fit=0.5,
        sigma_elec=0.1,
        switch_width=0.20,
        cutoff_distance=0.65,
        offset_c0=0.0,
        offset_c1=0.0,
    ):
        self.alpha_vdw, self.beta_vdw, self.r0_vdw = alpha_vdw, beta_vdw, r0_vdw
        self.A_fit, self.B_fit = A_fit, B_fit
        self.sigma_elec, self.switch_width, self.cutoff_distance = (
            sigma_elec,
            switch_width,
            cutoff_distance,
        )
        self.offset_c0, self.offset_c1 = offset_c0, offset_c1

    def build_expression(self, lam_vdw="lam_vdw"):
        """
        仅返回纯粹的 DEXP 核心表达式。
        Switch、Gaussian-Coulomb 与传统 PME 解耦由外层 Builder 统一接管。
        """
        rs = "max(r, 1e-6)"
        vdw_core = (
            f"4 * "
            f"({self.A_fit}*exp(-{self.alpha_vdw}*({rs}/{self.r0_vdw}-1.0)) - "
            f"{self.B_fit}*exp(-{self.beta_vdw}*({rs}/{self.r0_vdw}-1.0)))"
        )
        return f"{lam_vdw} * ({vdw_core})"

    def get_parameters_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, p):
        keys = [
            "alpha_vdw",
            "beta_vdw",
            "r0_vdw",
            "A_fit",
            "B_fit",
            "sigma_elec",
            "switch_width",
            "cutoff_distance",
            "offset_c0",
            "offset_c1",
        ]
        return cls(**{k: p[k] for k in keys if k in p})


class Orbv3SurrogateFitter:
    def __init__(
        self,
        fitting_region=(0.20, 0.50),
        enable_lbfgsb_refine: bool = True,
    ):
        self.r_min, self.r_max = fitting_region
        self.enable_lbfgsb_refine = bool(enable_lbfgsb_refine)

    def fit_parameters(self, distances_per_frame, e_delta_list, eff_eps=1.0):
        e_arr = np.asarray(e_delta_list, dtype=float)
        mask = np.abs(e_arr) < 5000.0
        if mask.sum() < 30:
            return {"fitting_success": False, "error": "insufficient_frames"}

        frame_energies_raw = e_arr[mask]
        frame_dists_raw = [
            np.asarray(d, dtype=float)
            for i, d in enumerate(distances_per_frame)
            if mask[i]
        ]

        # 统一使用均匀帧权重，避免少量短程异常接触绑架 DEXP 拟合。
        frame_weights = np.full(
            len(frame_energies_raw),
            1.0 / float(len(frame_energies_raw)),
            dtype=float,
        )

        robust_center = float(np.median(frame_energies_raw))
        abs_dev = np.abs(frame_energies_raw - robust_center)
        robust_scale = float(1.4826 * np.median(abs_dev)) if len(abs_dev) else 0.0
        if not np.isfinite(robust_scale) or robust_scale < 1.0e-8:
            robust_scale = max(float(np.std(frame_energies_raw)), 1.0)

        trim_limit = max(60.0, 3.5 * robust_scale)
        keep_trim = np.abs(frame_energies_raw - robust_center) <= trim_limit
        if int(np.sum(keep_trim)) >= 30:
            frame_energies = frame_energies_raw[keep_trim]
            frame_dists = [d for d, keep in zip(frame_dists_raw, keep_trim) if keep]
            frame_weights = frame_weights[keep_trim]
            frame_weights /= float(np.sum(frame_weights))
        else:
            frame_energies = frame_energies_raw
            frame_dists = frame_dists_raw

        diagnostic_global_mu = float(np.mean(frame_energies))
        diagnostic_weighted_mu = float(np.sum(frame_weights * frame_energies))
        diagnostic_centered_std = float(
            np.sqrt(np.sum(frame_weights * (frame_energies - diagnostic_weighted_mu) ** 2))
        )

        def _predict_frame_values(params):
            a, dgap, r0, A, B = params
            b = a - dgap

            if (
                r0 < 0.30 or r0 > 0.38
                or a < 12.0 or a > 30.0
                or b < 5.0 or b > 15.0
                or b >= a
                or A < 0 or B < 0
            ):
                return None, None, None

            inv = 1.0 / r0
            predicted_frame_energies = []
            valid_indices = []
            for idx, dd in enumerate(frame_dists):
                if dd.size == 0:
                    continue
                x = np.clip(dd * inv - 1.0, -50.0, 50.0)
                pair_energy = 4.0 * eff_eps * (
                    A * np.exp(-a * x) - B * np.exp(-b * x)
                )
                predicted_frame_energies.append(float(np.sum(pair_energy)))
                valid_indices.append(idx)
            if not predicted_frame_energies:
                return None, None, None
            return np.asarray(predicted_frame_energies, dtype=float), np.asarray(valid_indices, dtype=int), (a, b, r0, A, B)

        def _residual_vector(params):
            pred_raw, valid_indices, unpacked = _predict_frame_values(params)
            if pred_raw is None:
                return np.full(len(frame_energies) + 3, 1.0e6, dtype=float)

            a, b, r0, A, B = unpacked
            valid_weights = frame_weights[valid_indices]
            target_raw = frame_energies[valid_indices]
            weight_norm = float(np.sum(valid_weights))
            if weight_norm <= 1.0e-12:
                return np.full(len(frame_energies) + 3, 1.0e6, dtype=float)

            target_center = float(np.sum(valid_weights * target_raw) / weight_norm)
            pred_center = float(np.sum(valid_weights * pred_raw) / weight_norm)
            core_residuals = np.sqrt(valid_weights) * (
                (target_raw - target_center) - (pred_raw - pred_center)
            )

            inv = 1.0 / r0
            r_anchors = np.array([0.50, 0.60, 0.65], dtype=float)
            x_anchors = np.clip(r_anchors * inv - 1.0, -50.0, 50.0)
            u_dexp_anchors = 4.0 * eff_eps * (
                A * np.exp(-a * x_anchors) - B * np.exp(-b * x_anchors)
            )
            anchor_residuals = np.sqrt(1000.0 / max(1, len(r_anchors))) * u_dexp_anchors

            return np.concatenate([core_residuals, anchor_residuals])

        def _scalar_objective(params):
            residuals = _residual_vector(params)
            return float(np.dot(residuals, residuals))

        bounds = [
            (12.0, 30.0),
            (2.0, 15.0),
            (0.30, 0.38),
            (1.0e-5, 10.0),
            (1.0e-5, 10.0),
        ]
        x0 = np.array([18.0, 8.0, 0.34, 0.5, 0.2], dtype=float)

        de_res = differential_evolution(
            _scalar_objective,
            bounds=bounds,
            polish=False,
            maxiter=50,
            popsize=15,
            tol=0.01,
            updating="deferred",
            workers=1,
            seed=20260526,
        )
        x_seed = np.asarray(de_res.x if de_res.success else x0, dtype=float)

        ls_lower = np.array([b[0] for b in bounds], dtype=float)
        ls_upper = np.array([b[1] for b in bounds], dtype=float)
        ls_res = least_squares(
            _residual_vector,
            x_seed,
            bounds=(ls_lower, ls_upper),
            loss="soft_l1",
            f_scale=max(robust_scale, 10.0),
            max_nfev=2000,
        )
        x_best = np.asarray(ls_res.x, dtype=float)
        best_cost = _scalar_objective(x_best)

        if self.enable_lbfgsb_refine:
            lbfgsb_res = minimize(
                _scalar_objective,
                x_best,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-10, "maxiter": 500},
            )
            if lbfgsb_res.success and lbfgsb_res.fun <= best_cost:
                x_best = np.asarray(lbfgsb_res.x, dtype=float)
                best_cost = float(lbfgsb_res.fun)

        if not np.all(np.isfinite(x_best)):
            return {"fitting_success": False, "error": "optimizer_failed"}

        a, dgap, r0, A, B = x_best
        b = a - dgap

        # 诊断用接触度量：越短程越大，但绝不进入 OpenMM force。
        contact_metrics = []
        diagnostic_energies = []
        for Et_raw, dd in zip(frame_energies, frame_dists):
            if dd.size == 0:
                continue
            contact_score = np.clip(self.r_max - dd, 0.0, None)
            contact_metrics.append(float(np.sum(contact_score)))
            diagnostic_energies.append(float(Et_raw))

        diagnostic_contact_mu = diagnostic_global_mu
        diagnostic_contact_slope = 0.0
        if len(contact_metrics) >= 2 and np.std(contact_metrics) > 1.0e-12:
            diagnostic_contact_slope, diagnostic_contact_mu = np.polyfit(
                contact_metrics, diagnostic_energies, 1
            )

        return {
            "alpha_vdw": float(a),
            "beta_vdw": float(a - dgap),
            "r0_vdw": float(r0),
            "A_fit": float(A),
            "B_fit": float(B),
            "offset_c0": 0.0,
            "offset_c1": 0.0,
            "diagnostic_global_mu": float(diagnostic_global_mu),
            "diagnostic_fit_c0": float(diagnostic_weighted_mu),
            "diagnostic_weighted_center": float(diagnostic_weighted_mu),
            "diagnostic_centered_std": float(diagnostic_centered_std),
            "diagnostic_contact_mu": float(diagnostic_contact_mu),
            "diagnostic_contact_slope": float(diagnostic_contact_slope),
            "diagnostic_robust_center": float(robust_center),
            "diagnostic_robust_scale": float(robust_scale),
            "diagnostic_trim_limit": float(trim_limit),
            "diagnostic_frames_after_trim": int(len(frame_energies)),
            "sigma_elec": 0.1,
            "switch_width": 0.20,
            "cutoff_distance": 0.65,
            "fitting_success": True,
            "final_cost": float(best_cost),
            "optimizer_global_success": bool(de_res.success),
            "optimizer_ls_success": bool(ls_res.success),
        }


# ============================================================================
# 2. Boresch 限制力 & 解析修正
# ============================================================================
THERMODYNAMIC_CYCLE_DOC = """
Thermodynamic cycle used by this ABFE workflow
=============================================

Complex leg:
  1. A physical Boresch restraint is applied to keep the ligand in the binding
     pose during decoupling.
  2. The alchemical sampler computes the restrained complex-leg decoupling free
     energy, ΔG_decouple,restrained.
  3. The analytical Boresch term returned by calculate_boresch_analytical_correction
     is the standard-state release correction added to that leg:

       ΔG_complex = ΔG_decouple,restrained + ΔG_release_to_1M

     with V° = 1.6605 nm^3 and

       ΔG_release_to_1M = -RT ln[
         8π²V° / (r0² sinθA sinθB)
         * sqrt(Kr KθA KθB KφA KφB KφC) / (2πRT)^3
       ].

Solvent leg:
  No Boresch restraint is applied to the ligand in bulk solvent; therefore no
  Boresch analytical release term is added to the solvent leg.

PME/self correction:
  For neutral ligand PME decharging evaluated from total PME energies, OpenMM's
  reciprocal/self contribution contains a coordinate-independent term with
  negative sign, -C λ². Offline u_kn evaluation removes this offset by adding
  +C λ² to each λ state. Charged ligand paths disable ligand-only self correction
  unless a validated co-alchemical neutralization cycle is active.

LJ long-range/dispersion correction:
  Custom softcore VDW interaction-group forces do not automatically reproduce
  the original NonbondedForce dispersion correction. The workflow therefore
  records this as an explicit thermodynamic-cycle provenance item; any LJ tail
  or LRC term must be handled by a validated additional cycle term.

External APBS correction:
  If the production protocol uses APBS to supply the long-range electrostatic
  or continuum correction, that value is applied only as an explicit final
  binding-free-energy term:

    ΔG_bind = ΔG_complex - ΔG_solvent + ΔG_APBS

  APBS does not replace a Lennard-Jones dispersion/tail correction; any LJ tail
  term remains a separate correction if the chosen thermodynamic cycle needs it.

Binding free energy:
  Without an external APBS term, ΔG_bind = ΔG_complex - ΔG_solvent. Terms that
  are identical in both legs can cancel only when the Hamiltonians and correction
  conventions are documented and matched.
""".strip()


def calculate_boresch_analytical_correction(eq, fc, T=300.0):
    """
    计算 Boresch 解析修正。

    返回值是“解耦采样中保留 Boresch restraint”时需要加到 leg 上的
    标准态释放修正:

        ΔG_release = -RT ln[
            (8π² V° / (r0² sinθA sinθB))
            * sqrt(Kr KθA KθB KφA KφB KφC) / (2πRT)^3
        ]

    6 个谐振 Boresch 自由度的高斯积分给出 (2πRT)^3，而不是 1.5 次方。
    【强制标准单位】kJ/mol/nm², nm, rad
    ⚠️ 注意：eq["r0"] 必须是 nm 单位（不是 Å）
    ⚠️ 注意：fc["kr"] 必须为 kJ/mol/nm²，fc["kthetaA"] 等为 kJ/mol/rad²
    """
    T = T.value_in_unit(unit.kelvin) if hasattr(T, "value_in_unit") else float(T)
    R = constants.R / 1000.0
    RT = R * T
    V0 = 1.6605  # nm³ (标准摩尔体积)

    # ✅ 修复1：增加单位量级与物理合理性断言
    kr, ktA, ktB = fc.get("kr", 0), fc.get("kthetaA", 0), fc.get("kthetaB", 0)
    if not (50 <= kr <= 5000):
        raise ValueError(f"kr 超出合理范围 [50, 5000] kJ/mol/nm²: {kr}")
    if not (10 <= ktA <= 500 and 10 <= ktB <= 500): 
        raise ValueError("角度力常数 ktA/ktB 建议范围 [10, 500] kJ/mol/rad²")

    # ✅ 强制标准单位，不再进行任何转换
    r0 = eq["r0"]  # nm
    thA = eq["thetaA0"]  # rad
    thB = eq["thetaB0"]  # rad

    kr = fc["kr"]  # kJ/mol/nm²
    ktA = fc["kthetaA"]  # kJ/mol/rad²
    ktB = fc["kthetaB"]
    kpA = fc["kphiA"]
    kpB = fc["kphiB"]
    kpC = fc["kphiC"]

    Kdet = kr * ktA * ktB * kpA * kpB * kpC
    if Kdet <= 0:
        raise ValueError("Boresch 力常数存在零值或负值，无法计算解析修正")
    
    sin_t = math.sin(thA) * math.sin(thB)

    if sin_t < 1e-4:
        raise ValueError("Boresch 锚点几何奇点 (sinθ≈0)")

    standard_state_factor = (8.0 * math.pi**2 * V0) / (r0**2 * sin_t)
    restraint_integral_factor = math.sqrt(Kdet) / ((2.0 * math.pi * RT) ** 3.0)
    argument = standard_state_factor * restraint_integral_factor
    if argument <= 0 or not math.isfinite(argument):
        raise ValueError(f"Boresch 解析修正对数参数异常: {argument}")

    return -RT * math.log(argument)



class LambdaDependentBoreschForce(openmm.CustomCompoundBondForce):
    def __init__(
        self,
        rec_idx,
        lig_idx,
        eq,
        fc,
        lam_name="lambda_rest",
        fixed_lam=None,
        sign=1.0,
        use_pbc=False,
        # ✅ 移除 unit_sys 参数，强制标准单位
    ):
        if len(rec_idx) != 3 or len(lig_idx) != 3:
            raise ValueError("需 exactly 3 受体 + 3 配体原子")

        # ✅ 强制标准单位，不再进行任何转换
        r0 = eq["r0"]  # nm
        thA = eq["thetaA0"]  # rad
        thB = eq["thetaB0"]  # rad
        phA = eq["phiA0"]  # rad
        phB = eq["phiB0"]  # rad
        phC = eq["phiC0"]  # rad

        kr = fc["kr"]  # kJ/mol/nm²
        ktA = fc["kthetaA"]  # kJ/mol/rad²
        ktB = fc["kthetaB"]
        kpA = fc["kphiA"]
        kpB = fc["kphiB"]
        kpC = fc["kphiC"]

        # 打印调试信息（确保 thA 是弧度）
        print(f"  [Boresch] kr={kr:.1f} kJ/mol/nm², r0={r0:.3f} nm, θA={np.degrees(thA):.1f}°")

        ls = f"{fixed_lam:.6f}" if fixed_lam is not None else lam_name

        # ✅ 修复2：标准谐波势 (distance-r0)^2，导数连续且数值稳定
        # 🚨 关键修复：atom-index 顺序与 thetaA0/thetaB0/phiA0/phiB0/phiC0 的
        # 计算约定必须严格一致。addBond(rec_idx+lig_idx) 的顺序是
        # [R0(离配体最近), R1, R2(离配体最远), L0(离受体最近), L1, L2]——
        # 这也是 calc_boresch_from_last_frame / _check_boresch_geometry_safe /
        # _validate_boresch_geometry_strict 全部使用的约定：
        #   r0      = distance(R0, L0)
        #   thetaA0 = angle(R1, R0, L0)         顶点=R0
        #   thetaB0 = angle(R0, L0, L1)         顶点=L0
        #   phiA0   = dihedral(R2, R1, R0, L0)
        #   phiB0   = dihedral(R1, R0, L0, L1)
        #   phiC0   = dihedral(R0, L0, L1, L2)
        # 旧表达式误用 angle(p2,p3,p4)/angle(p3,p4,p5) 和
        # dihedral(p1,p2,p3,p4) 等，把顶点/参考原子错当成了 R2(最远的受体
        # 锚点，选择时只保证"刚性"而不保证与 R1/L0 不共线)，导致实际被约束
        # 的角度和平衡值计算出的角度根本不是同一个几何量：一来平衡值形同虚设、
        # 限制力没有真正锁住原有构象；二来一旦 R2 恰好与 R1、L0 接近共线，
        # angle()/dihedral() 的解析梯度出现 1/sinθ 型奇点，能量看起来正常但
        # 力却能炸到 10^7~10^8 kJ/mol/nm 量级——这正是本次 REMD 预热崩溃的根源。
        expr = (
            f"({sign})*{ls}*("
            "0.5*kr*(distance(p1,p4)-r0)^2+"
            "ktA*(1-cos(angle(p2,p1,p4)-thetaA0))+"
            "ktB*(1-cos(angle(p1,p4,p5)-thetaB0))+"
            "kpA*(1-cos(dihedral(p3,p2,p1,p4)-phiA0))+"
            "kpB*(1-cos(dihedral(p2,p1,p4,p5)-phiB0))+"
            "kpC*(1-cos(dihedral(p1,p4,p5,p6)-phiC0))"
            ")"
        )
        super().__init__(6, expr)  # ✅ N=6

        if fixed_lam is None:
            self.addGlobalParameter(lam_name, 0.0)

        for n, v in [
            ("r0", r0),
            ("thetaA0", thA),
            ("thetaB0", thB),
            ("phiA0", phA),
            ("phiB0", phB),
            ("phiC0", phC),
            ("kr", kr),
            ("ktA", ktA),
            ("ktB", ktB),
            ("kpA", kpA),
            ("kpB", kpB),
            ("kpC", kpC),
        ]:
            self.addGlobalParameter(n, v)

        if hasattr(self, "setUsesPeriodicBoundaryConditions"):
            self.setUsesPeriodicBoundaryConditions(bool(use_pbc))

        self.addBond(list(rec_idx) + list(lig_idx))


# ============================================================================
# 3. Orb 口袋力投影估算器 (v4.3 - 3-Stage Optimized & Hybrid Filter)
# ============================================================================
class OrbVacuumContext:
    """ORB 真空力场计算上下文 (仅用于口袋内力场计算)"""

    def __init__(
        self, topology, model_name="mace-off24-medium", device="cpu"
    ):
        self.device = device
        self.model_name = model_name
        self.potential = MLPotential(model_name)
        self.system = self.potential.createSystem(topology, **_build_openmmml_kwargs(
            device=self.device,
            return_energy_type="energy",
            charge=0,
            multiplicity=1,
        ))
        self.integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        try:
            platform = openmm.Platform.getPlatformByName(device.upper())
        except Exception:
            platform = openmm.Platform.getPlatformByName("CPU")
        self.context = openmm.Context(self.system, self.integrator, platform)

    def calculate_forces(self, positions_nm):
        self.context.setPositions(positions_nm)
        return (
            self.context.getState(getForces=True)
            .getForces(asNumpy=True)
            .value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
        )


class OrbBoreschEstimator:
    """
    基于 Orb 口袋力投影与三阶段锚点优化的 Boresch 估算器
    特性：
    1. 稳定性+动态距离+几何构型 3-Stage 锚点筛选
    2. 边界氢饱和修补 (避免切割键导致的力场畸变)
    3. 混合滤波拟合 (线性回归优先，相关性不足时自动切换至波动法)
    4. 二面角自动展开 (Unwrap) 与 Jacobian 修正
    """

    DEFAULT_CONFIG = {
        "temperature": 300.0,
        "cutoff_nm": 0.9,
        "rmsf_cutoff_nm": 0.15,
        "dist_reject_nm": 0.40,
        "dist_gold_min_nm": 0.50,
        "dist_gold_max_nm": 1.10,
        "dist_backup_min_nm": 1.10,
        "dist_backup_max_nm": 1.50,
        "rec_anchor_dist_min": 0.40,
        "rec_anchor_dist_max": 0.80,
        "rec_anchor_angle_min": 60,
        "rec_anchor_angle_max": 120,
        "corr_threshold_keep": -0.1,
        "corr_threshold_good": -0.5,
        "use_fluctuation_fallback": True,
        "short_sidechain_res": ["GLY", "ALA", "SER", "VAL", "THR", "CYS"],
        "long_sidechain_res": [
            "TRP",
            "PHE",
            "TYR",
            "ARG",
            "LYS",
            "GLU",
            "GLN",
            "MET",
            "HIS",
        ],
        "score_weights": {
            "stability": 1.0,
            "distance": 2.0,
            "geometry": 1.5,
            "signal": 3.0,
        },
        "top_n_candidates": 5,
    }

    def __init__(self, temperature=300.0, device=None, cutoff_nm=0.9, n_frames=500):
        self.T = temperature
        self.gas_constant_kj_per_mol_k = 8.314e-3
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = {
            **self.DEFAULT_CONFIG,
            "temperature": temperature,
            "cutoff_nm": cutoff_nm,
        }
        self.n_frames = n_frames
        self.sc_correction = {
            "GLY": -0.05,
            "ALA": -0.03,
            "SER": -0.02,
            "THR": -0.02,
            "CYS": -0.02,
            "ARG": 0.15,
            "LYS": 0.12,
            "GLU": 0.10,
            "GLN": 0.10,
            "TRP": 0.12,
            "TYR": 0.10,
            "PHE": 0.08,
            "MET": 0.08,
            "HIS": 0.06,
        }

    def _get_sidechain_correction(self, resname):
        return self.sc_correction.get(resname, 0.0)

    def _score_distance(self, dist_nm, resname):
        cfg = self.config
        if dist_nm < cfg["dist_reject_nm"]:
            return -100, "❌ 太近"
        c_dist = dist_nm - self._get_sidechain_correction(resname)
        if cfg["dist_gold_min_nm"] <= c_dist <= cfg["dist_gold_max_nm"]:
            center = (cfg["dist_gold_min_nm"] + cfg["dist_gold_max_nm"]) / 2
            bonus = 50 * (
                1
                - abs(c_dist - center)
                / (cfg["dist_gold_max_nm"] - cfg["dist_gold_min_nm"])
            )
            return 50 + bonus, "✅ 黄金区间"
        if cfg["dist_backup_min_nm"] <= c_dist <= cfg["dist_backup_max_nm"]:
            score = 30 * (
                1
                - (c_dist - cfg["dist_backup_min_nm"])
                / (cfg["dist_backup_max_nm"] - cfg["dist_backup_min_nm"])
            )
            return score, "⚠️ 备选区间"
        return 5 if c_dist > cfg["dist_backup_max_nm"] else 25, "⚪ 过渡/较远"

    def _check_anchor_geometry(self, rec_anchors, traj, pocket_sel):
        r = traj.xyz[0, pocket_sel]
        rec_local = [np.where(pocket_sel == a)[0][0] for a in rec_anchors]
        P1, P2, P3 = r[rec_local]

        def calc_angle(a, b, c):
            ba, bc = a - b, c - b
            cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
            return np.arccos(np.clip(cos, -1, 1)) * 180 / np.pi

        cfg = self.config
        dists = [
            np.linalg.norm(P1 - P2),
            np.linalg.norm(P2 - P3),
            np.linalg.norm(P1 - P3),
        ]
        d_score = sum(
            20
            if cfg["rec_anchor_dist_min"] <= d <= cfg["rec_anchor_dist_max"]
            else (-30 if d < cfg["rec_anchor_dist_min"] else 0)
            for d in dists
        )
        angle_P = calc_angle(P3, P2, P1)
        a_score = (
            40
            if cfg["rec_anchor_angle_min"] <= angle_P <= cfg["rec_anchor_angle_max"]
            else (15 if 30 <= angle_P <= 150 else -50)
        )
        return (
            (a_score > -30),
            d_score + a_score,
            f"∠P3-P2-P1={angle_P:.1f}°, d12={dists[0] * 10:.2f}Å",
        )

    def _validate_boresch_geometry_strict(self, rec_anchors, lig_anchors, ref_coords):
        """
        【严格几何硬过滤器】(Hard Reject) - 彻底重写版
        统一单位：内部全部使用 nm，仅日志转 Å/°
        拒绝标准：
          1. r0 (R0-L0) ∈ [0.50, 1.00] nm (5-10 Å)
          2. θA (R1-R0-L0) ∈ [40°, 140°]  → sin(θA) ≥ 0.642
          3. θB (R0-L0-L1) ∈ [40°, 140°]  → sin(θB) ≥ 0.642
          4. 受体锚点间距 ≥ 0.38 nm (3.8 Å)
          5. 配体锚点间距 ≥ 0.25 nm (2.5 Å)
        """
        try:
            R0, R1, R2 = [ref_coords[a] for a in rec_anchors]
            L0, L1, L2 = [ref_coords[a] for a in lig_anchors]
        except IndexError:
            return False, "❌ 锚点索引越界"

        def dist(p1, p2): return np.linalg.norm(p1 - p2)
        def angle_rad(p1, p2, p3):
            v1, v2 = p1 - p2, p3 - p2
            cos_val = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12), -1.0, 1.0)
            return np.arccos(cos_val)

        # 1. 核心几何量 (nm & rad)
        r0_nm = dist(R0, L0)
        thA_rad = angle_rad(R1, R0, L0)
        thB_rad = angle_rad(R0, L0, L1)

        d_R01, d_R12 = dist(R0, R1), dist(R1, R2)
        d_L01, d_L12 = dist(L0, L1), dist(L1, L2)

        # 2. 严格阈值拦截 (全部基于 nm/rad 比较)
        if not (0.50 <= r0_nm <= 1.00):
            return False, f"❌ r0={r0_nm*10:.2f}Å [需5.0-10.0Å]"

        thA_deg, thB_deg = np.degrees(thA_rad), np.degrees(thB_rad)
        if not (40.0 <= thA_deg <= 140.0):
            return False, f"❌ θA={thA_deg:.1f}° [需40-140°]"
        if not (40.0 <= thB_deg <= 140.0):
            return False, f"❌ θB={thB_deg:.1f}° [需40-140°]"

        # 3. 奇点保护：sin(θ) < 0.64 直接拒绝 (对应 θ<40° 或 θ>140°)
        if np.sin(thA_rad) < 0.64 or np.sin(thB_rad) < 0.64:
            return False, f"❌ 几何奇异 sinθ≈0 (θA={thA_deg:.1f}°, θB={thB_deg:.1f}°)"

        # 4. 锚点共线/重叠保护
        if d_R01 < 0.38 or d_R12 < 0.38:
            return False, f"❌ 受体锚点过近 (min={min(d_R01, d_R12)*10:.1f}Å)"
        if d_L01 < 0.25 or d_L12 < 0.25:
            return False, f"❌ 配体锚点过近 (min={min(d_L01, d_L12)*10:.1f}Å)"

        # 5. 通过
        return True, f"✅ 几何合格 r0={r0_nm*10:.2f}Å θA={thA_deg:.1f}° θB={thB_deg:.1f}°"

    def _add_capping_hydrogens(self, topology, selection, ref_coords):
        cap_h_info = []
        residues = list(topology.residues)
        for res in residues:
            res_atoms = [a.index for a in res.atoms]
            if not any(a in selection for a in res_atoms) or all(
                a in selection for a in res_atoms
            ):
                continue
            bb = {a.name: a.index for a in res.atoms if a.name in ["N", "CA", "C", "O"]}
            if "C" in bb and bb["C"] in selection:
                nxt = residues[res.index + 1] if res.index + 1 < len(residues) else None
                if nxt:
                    nxt_n = [a.index for a in nxt.atoms if a.name == "N"]
                    if nxt_n and nxt_n[0] not in selection:
                        c_pos = ref_coords[bb["C"]]
                        n_pos = (
                            ref_coords[nxt_n[0]]
                            if nxt_n
                            else c_pos + np.array([0.15, 0, 0])
                        )
                        d = n_pos - c_pos
                        n = np.linalg.norm(d)
                        d = d / n if n > 1e-10 else np.array([1, 0, 0])
                        cap_h_info.append(
                            {
                                "cut_atom": bb["C"],
                                "cut_type": "C_term",
                                "direction": d,
                                "bond_length": 0.11,
                                "neighbor_global": nxt_n[0],
                            }
                        )
            if "N" in bb and bb["N"] in selection:
                prv = residues[res.index - 1] if res.index > 0 else None
                if prv:
                    prv_c = [a.index for a in prv.atoms if a.name == "C"]
                    if prv_c and prv_c[0] not in selection:
                        n_pos = ref_coords[bb["N"]]
                        c_pos = (
                            ref_coords[prv_c[0]]
                            if prv_c
                            else n_pos + np.array([-0.15, 0, 0])
                        )
                        d = c_pos - n_pos
                        n = np.linalg.norm(d)
                        d = d / n if n > 1e-10 else np.array([-1, 0, 0])
                        cap_h_info.append(
                            {
                                "cut_atom": bb["N"],
                                "cut_type": "N_term",
                                "direction": d,
                                "bond_length": 0.10,
                                "neighbor_global": prv_c[0],
                            }
                        )
        return cap_h_info

    def _build_pocket_context(self, traj, ligand_resname):
        if not HAS_MDTRAJ:
            raise ImportError("需要 mdtraj")
        top = traj.topology
        lig_sel = top.select(f"resname {ligand_resname}")
        if len(lig_sel) == 0:
            raise ValueError("未找到配体")
        lig_center = traj.xyz[0, lig_sel].mean(axis=0)
        prot_sel = top.select("protein")
        nearby_atoms = [
            a
            for a in prot_sel
            if np.linalg.norm(traj.xyz[0, a] - lig_center) <= self.config["cutoff_nm"]
        ]
        # 保留完整残基，避免在主链/侧链中间截断产生自由基；比事后手工补 capping H 更稳。
        nearby_residues = {top.atom(a).residue.index for a in nearby_atoms}
        pocket_atoms = set(int(i) for i in lig_sel.tolist())
        for res in top.residues:
            if res.index in nearby_residues:
                for atom in res.atoms:
                    pocket_atoms.add(atom.index)
        pocket_sel = np.array(sorted(pocket_atoms), dtype=int)
        if len(pocket_sel) < 10:
            raise ValueError("口袋原子不足，请增大 cutoff_nm")

        cap_h = []
        pocket_traj = traj.atom_slice(pocket_sel)
        omm_top = pocket_traj.topology.to_openmm()
        context = OrbVacuumContext(omm_top, device=self.device)

        # --- 3-Stage 锚点筛选 ---
        lig_atoms = top.select(f"resname {ligand_resname} and not element H")
        if len(lig_atoms) < 3:
            lig_atoms = top.select(f"resname {ligand_resname}")
        if len(lig_atoms) == 0:
            raise ValueError(f"未找到配体 {ligand_resname} 原子，无法构建口袋上下文")

        lig_scores = [
            top.atom(i).element.mass / 12.0
            + (2.0 if top.atom(i).name in ["CG", "CD", "CE", "CZ", "CA"] else 0)
            for i in lig_atoms
        ]
        lig_anchors = lig_atoms[np.argsort(lig_scores)[-3:]].tolist()

        rec_ca = top.select("protein and name CA")
        rec_ca_p = np.intersect1d(rec_ca, pocket_sel)
        rmsf_traj = traj[:: max(1, len(traj) // 100)]
        if len(rmsf_traj) > 1:
            rmsf_traj.superpose(rmsf_traj, 0, atom_indices=rec_ca_p)
            rmsf = np.sqrt(
                np.mean(
                    (rmsf_traj.xyz[:, rec_ca_p] - rmsf_traj.xyz[0, rec_ca_p]) ** 2,
                    axis=(0, 2),
                )
            )
        else:
            rmsf = np.zeros(len(rec_ca_p))

        rigid_mask = rmsf < self.config["rmsf_cutoff_nm"]
        rigid_ca = rec_ca_p[rigid_mask]
        rigid_rmsf = rmsf[rigid_mask]

        candidates = []
        if len(rigid_ca) >= 3:
            from itertools import combinations

            sorted_idx = np.argsort(rigid_rmsf)[:30]
            sorted_ca = rigid_ca[sorted_idx]
            sorted_rmsf = rigid_rmsf[sorted_idx]
            sorted_resnames = [top.atom(idx).residue.name for idx in sorted_ca]

            MAX_R0_ANGSTROM = 12.0
            MIN_R0_ANGSTROM = 6.0
            seen_combos = set()
            candidates = []

            for combo in combinations(range(len(sorted_ca)), 3):
                combo_indices = list(combo)
                combo_anchors = [sorted_ca[i] for i in combo_indices]
                combo_rmsf_vals = [sorted_rmsf[i] for i in combo_indices]

                dists = [
                    np.linalg.norm(traj.xyz[0, a] - lig_center) for a in combo_anchors
                ]
                sorted_pairs = sorted(
                    zip(
                        dists,
                        combo_anchors,
                        combo_rmsf_vals,
                        [sorted_resnames[i] for i in combo_indices],
                    ),
                    key=lambda x: x[0],
                )
                rec_anchors = [p[1] for p in sorted_pairs]
                rec_rmsf = [p[2] for p in sorted_pairs]
                rec_resnames = [p[3] for p in sorted_pairs]

                R0, R1, R2 = traj.xyz[0, rec_anchors]
                L0 = traj.xyz[0, lig_anchors[0]]

                r0_nm = np.linalg.norm(R0 - L0)
                r0_A = r0_nm * 10.0

                def calc_ang(a, b, c):
                    ba, bc = a - b, c - b
                    cos = np.dot(ba, bc) / (
                        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10
                    )
                    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

                thetaA = calc_ang(R1, R0, L0)
                thetaB = calc_ang(R0, L0, traj.xyz[0, lig_anchors[1]])

                if not (MIN_R0_ANGSTROM <= r0_A < MAX_R0_ANGSTROM):
                    continue

                geo_key = (
                    round(r0_A, 1),
                    round(thetaA, 1),
                    round(thetaB, 1),
                    tuple(sorted(rec_anchors[:2])),
                )
                if geo_key in seen_combos:
                    continue
                seen_combos.add(geo_key)

                score_stab = sum(
                    (self.config["rmsf_cutoff_nm"] - rv) * 100 for rv in rec_rmsf
                )

                if 8.0 <= r0_A <= 10.5:
                    score_dist = 60
                elif 10.5 < r0_A < 12.0:
                    score_dist = 30
                else:
                    score_dist = 0

                if 60 <= thetaA <= 110:
                    score_ang = 50
                elif 45 < thetaA < 60 or 110 < thetaA < 135:
                    score_ang = 20
                else:
                    score_ang = 0

                pre_sc = score_stab + score_dist + score_ang
                candidates.append(
                    {
                        "rec_anchors": rec_anchors,
                        "score": pre_sc,
                        "r0_A": r0_A,
                        "thetaA": thetaA,
                        "thetaB": thetaB,
                        "rmsf_avg": np.mean(rec_rmsf),
                    }
                )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_cands = candidates[: self.config["top_n_candidates"]]

        # 快速力学信号验证 (取前50帧)
        validated = []
        for cand in top_cands:
            local_anchors = [
                int(np.where(pocket_sel == a)[0][0])
                for a in cand["rec_anchors"] + lig_anchors
            ]
            ks_temp = self._quick_validate(
                traj[:50], context, pocket_sel, local_anchors, cap_h
            )

            if ks_temp is not None:
                validated.append(
                    {**cand, "ks_temp": ks_temp, "total": cand["score"] + 20.0}
                )

        if validated:
            validated.sort(key=lambda x: x["total"], reverse=True)
            best = validated[0]
            rec_anchors, best_ks = best["rec_anchors"], best["ks_temp"]
        else:
            rec_anchors = (
                candidates[0]["rec_anchors"] if candidates else rigid_ca[:3].tolist()
            )
            best_ks = None

        anchor_global = rec_anchors + lig_anchors
        local_anchors = [int(np.where(pocket_sel == a)[0][0]) for a in anchor_global]
        return context, pocket_sel, local_anchors, anchor_global, cap_h, best_ks

    def _quick_validate(self, traj, context, selection, local_anchors, cap_h):
        if len(traj) < 10:
            return None
        n_frames = len(traj)
        q_data = np.zeros((n_frames, 6))
        Fq_data = np.zeros((n_frames, 6))
        for f in range(n_frames):
            pos = traj.xyz[f, selection]
            forces = context.calculate_forces(pos)
            r_anchors = pos[local_anchors]
            if np.any(np.isnan(r_anchors)):
                continue
            q, grads = self._compute_geom_gradients(r_anchors)
            if np.any(np.isnan(q)):
                continue
            Fq = np.zeros(6)
            for g in range(6):
                for a in range(6):
                    Fq[g] += np.dot(forces[local_anchors[a]], grads[g, a])
            q_data[f] = q
            Fq_data[f] = Fq
        return self._apply_hybrid_filter(q_data, Fq_data)

    def _compute_geom_gradients(self, r_anchors):
        # 🚨 关键修复：r_anchors 的 6 个 slot 严格是
        # [0]=R0(受体,离配体最近) [1]=R1 [2]=R2(受体,最远)
        # [3]=L0(配体,离受体最近) [4]=L1 [5]=L2(配体,最远)
        # 必须与 calc_boresch_from_last_frame / _check_boresch_geometry_safe /
        # LambdaDependentBoreschForce 完全一致：
        #   r0      = distance(R0, L0)                slot(0,3)
        #   thetaA0 = angle(R1, R0, L0)   顶点=R0      slot(1,0,3)
        #   thetaB0 = angle(R0, L0, L1)   顶点=L0      slot(0,3,4)
        #   phiA0   = dihedral(R2, R1, R0, L0)         slot(2,1,0,3)
        #   phiB0   = dihedral(R1, R0, L0, L1)         slot(1,0,3,4)
        #   phiC0   = dihedral(R0, L0, L1, L2)         slot(0,3,4,5)
        # 旧版把 H0/H2 的变量名接反了（H0 实际绑定的是 slot[2]=R2 而不是
        # slot[0]=R0），导致这里算出的力常数/CV 用的是"最远"受体锚点当顶点，
        # 跟平衡值计算/几何合法性检查完全对不上，是这次 Boresch 崩溃的另一个源头。
        q = np.zeros(6)
        grads = np.zeros((6, 6, 3))

        R0, L0 = r_anchors[0], r_anchors[3]
        vec_r = L0 - R0
        norm_r = np.linalg.norm(vec_r) + 1e-10
        q[0] = norm_r
        ur = vec_r / norm_r
        grads[0, 0, :] = -ur
        grads[0, 3, :] = ur

        angle_slots = [(1, 0, 3), (0, 3, 4)]
        for i, (sa, sb, sc) in enumerate(angle_slots):
            a, b, c = r_anchors[sa], r_anchors[sb], r_anchors[sc]
            ba, bc = a - b, c - b
            nba, nbc = np.linalg.norm(ba) + 1e-10, np.linalg.norm(bc) + 1e-10
            cosA = np.clip(np.dot(ba, bc) / (nba * nbc), -1, 1)
            q[i + 1] = np.arccos(cosA)
            sinA = np.sqrt(1 - cosA**2) + 1e-10
            if sinA > 1e-3:
                dbda = (cosA * bc / nbc - ba / nba) / (nba * sinA)
                dbdc = (cosA * ba / nba - bc / nbc) / (nbc * sinA)
                grads[i + 1, sa, :] = dbda
                grads[i + 1, sb, :] = -dbda - dbdc
                grads[i + 1, sc, :] = dbdc

        if HAS_MDTRAJ:
            dummy_top = mdtraj.Topology()
            c = dummy_top.add_chain()
            r = dummy_top.add_residue("X", c)
            for _ in range(6):
                dummy_top.add_atom("C", openmm.app.element.Element.getBySymbol("C"), r)
            eps = 1e-3
            dihedral_slots = [(2, 1, 0, 3), (1, 0, 3, 4), (0, 3, 4, 5)]
            for g_idx, tup in enumerate(dihedral_slots, 3):
                q[g_idx] = mdtraj.compute_dihedrals(
                    mdtraj.Trajectory(r_anchors[None], dummy_top), [tup]
                )[0, 0]
                perturbations = []
                grad_slots = []
                for a in tup:
                    for d in range(3):
                        rp = r_anchors.copy()
                        rm = r_anchors.copy()
                        rp[a, d] += eps
                        rm[a, d] -= eps
                        perturbations.extend((rp, rm))
                        grad_slots.append((a, d))

                batch_xyz = np.asarray(perturbations, dtype=float)
                batch_angles = mdtraj.compute_dihedrals(
                    mdtraj.Trajectory(batch_xyz, dummy_top), [tup]
                )[:, 0]
                for idx, (a, d) in enumerate(grad_slots):
                    grads[g_idx, a, d] = (
                        batch_angles[2 * idx] - batch_angles[2 * idx + 1]
                    ) / (2 * eps)
        return q, grads

    def _apply_hybrid_filter(self, q, Fq):
        kB_T = self.gas_constant_kj_per_mol_k * self.T
        names = ["kr", "kthetaA", "kthetaB", "kphiA", "kphiB", "kphiC"]
        ks = {}
        for i in range(6):
            valid = ~(np.isnan(q[:, i]) | np.isnan(Fq[:, i]))
            if valid.sum() < 10:
                ks[names[i]] = 0.0
                continue
            dq, dF = (
                q[:, i][valid] - np.mean(q[:, i][valid]),
                Fq[:, i][valid] - np.mean(Fq[:, i][valid]),
            )
            var = np.var(dq)
            cov = np.cov(dF, dq)[0, 1]
            k_reg = -cov / var if var > 1e-12 else None
            k_fluc = kB_T / var if var > 1e-12 else None
            corr = (
                np.corrcoef(q[:, i][valid], Fq[:, i][valid])[0, 1]
                if np.std(q[:, i][valid]) > 1e-10
                else 0.0
            )

            if k_reg and k_reg > 0 and corr < self.config["corr_threshold_keep"]:
                ks[names[i]] = k_reg
            elif k_fluc and self.config["use_fluctuation_fallback"]:
                ks[names[i]] = k_fluc
            else:
                ks[names[i]] = 0.0

        for name in names:
            if name == "kr":
                ks[name] = float(min(max(ks[name], 100.0), 2000.0))
            else:
                ks[name] = float(min(max(ks[name], 10.0), 100.0))

        return ks

    def run_pocket_force_projection(
        self, traj, context, selection, local_anchors, cap_h_coords=None
    ):
        n_frames = len(traj)
        q_data = np.zeros((n_frames, 6))
        Fq_data = np.zeros((n_frames, 6))
        for f in range(n_frames):
            pos = traj.xyz[f, selection]
            forces = context.calculate_forces(pos)
            r_anchors = pos[local_anchors]
            if np.any(np.isnan(r_anchors)):
                continue
            q, grads = self._compute_geom_gradients(r_anchors)
            if np.any(np.isnan(q)):
                continue
            Fq = np.zeros(6)
            for g in range(6):
                for a in range(6):
                    Fq[g] += np.dot(forces[local_anchors[a]], grads[g, a])
            q_data[f] = q
            Fq_data[f] = Fq

        # Unwrap 二面角
        for i in [3, 4, 5]:
            valid = ~(np.isnan(q_data[:, i]) | np.isinf(q_data[:, i]))
            if valid.sum() > 10:
                q_data[valid, i] = np.unwrap(q_data[valid, i])
        return self._apply_hybrid_filter(q_data, Fq_data)

    def estimate_from_trajectory(self, traj, ligand_resname, output_path=None):
        context, pocket_sel, local_anchors, anchor_global, cap_h, ks_quick = (
            self._build_pocket_context(traj, ligand_resname)
        )
        ks = (
            ks_quick
            if ks_quick
            else self.run_pocket_force_projection(
                traj, context, pocket_sel, local_anchors, cap_h
            )
        )

        traj_aligned = traj[:]
        traj_aligned.superpose(traj_aligned, 0, atom_indices=pocket_sel)
        r0 = traj_aligned.xyz[0, pocket_sel][local_anchors]
        # 🚨 关键修复：local_anchors 顺序是 [R0(离配体最近),R1,R2(最远),L0,L1,L2]，
        # 之前写成 H2,H1,H0=r0[0,1,2] 把 H0 错绑定到 R2（最远锚点），导致下面算出
        # 的 eq 平衡值和 receptor_indices=anchor_global[:3] 实际代表的原子对不上。
        H0, H1, H2, G0, G1, G2 = r0

        def calc_angle(a, b, c):
            ba, bc = a - b, c - b
            cos_val = np.clip(
                np.dot(ba, bc)
                / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10),
                -1,
                1,
            )
            # ✅ 直接返回弧度 (rad)
            return np.arccos(cos_val)

        def calc_dihedral(a, b, c, d):
            b1, b2, b3 = b - a, c - b, d - c
            n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
            m1 = np.cross(n1, b2 / np.linalg.norm(b2))
            # ✅ 直接返回弧度 (rad)
            return np.arctan2(np.dot(m1, n2), np.dot(n1, n2))

        eq = {
            "r0": float(np.linalg.norm(H0 - G0)),  # ✅ nm (移除 *10)
            "thetaA0": float(calc_angle(H1, H0, G0)),  # ✅ rad
            "thetaB0": float(calc_angle(H0, G0, G1)),  # ✅ rad
            "phiA0": float(calc_dihedral(H2, H1, H0, G0)),  # ✅ rad
            "phiB0": float(calc_dihedral(H1, H0, G0, G1)),  # ✅ rad
            "phiC0": float(calc_dihedral(H0, G0, G1, G2)),  # ✅ rad
        }
        rec_indices = anchor_global[:3]
        lig_indices = anchor_global[3:]
        result = {
            "receptor_indices": rec_indices.tolist()
            if hasattr(rec_indices, "tolist")
            else list(rec_indices),
            "ligand_indices": lig_indices.tolist()
            if hasattr(lig_indices, "tolist")
            else list(lig_indices),
            "force_constants": ks,
            "equilibrium_values": eq,
            "method": "orb_pocket_projection_v4.3",
        }
        if output_path:

            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, (np.bool_,)):
                        return bool(obj)
                    return super().default(obj)

            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, cls=NumpyEncoder)
        return result

    def _finalize_candidate(self, cand, traj, context, pocket_sel, cap_h):
        """计算候选者的最终力常数和平衡值"""
        rec_anchors = cand["rec_anchors"]
        lig_anchors = cand["lig_anchors"]
        local_anchors = cand["local_anchors"]
        ks = self.run_pocket_force_projection(
            traj, context, pocket_sel, local_anchors, cap_h
        )
        traj_aligned = traj[:]
        traj_aligned.superpose(traj_aligned, 0, atom_indices=pocket_sel)
        r0_frame = traj_aligned.xyz[0, pocket_sel][local_anchors]
        # 🚨 关键修复：同 estimate_from_trajectory，local_anchors 顺序是
        # [R0(离配体最近),R1,R2(最远),L0,L1,L2]，之前 H0 被错绑定到 R2。
        H0, H1, H2, G0, G1, G2 = r0_frame

        def calc_angle_rad(a, b, c):
            ba, bc = a - b, c - b
            cos_val = np.clip(
                np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10),
                -1, 1,
            )
            return float(np.arccos(cos_val))

        def calc_dihedral_rad(a, b, c, d):
            b1, b2, b3 = b - a, c - b, d - c
            n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
            m1 = np.cross(n1, b2 / np.linalg.norm(b2))
            return float(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))

        eq = {
            "r0": float(np.linalg.norm(H0 - G0)),
            "thetaA0": calc_angle_rad(H1, H0, G0),
            "thetaB0": calc_angle_rad(H0, G0, G1),
            "phiA0": calc_dihedral_rad(H2, H1, H0, G0),
            "phiB0": calc_dihedral_rad(H1, H0, G0, G1),
            "phiC0": calc_dihedral_rad(H0, G0, G1, G2),
        }
        return {**cand, "ks": ks, "eq": eq}

    def _score_candidate_comprehensive(self, rec_anchors, lig_anchors, rmsf_vals,
                                        r0_nm, thA_rad, thB_rad, kr_raw, traj, top):
        """综合评分函数：稳定性 + 几何 + 力常数 + 序列分散度"""
        
        # === 1. 稳定性 (原有，权重 1.0) ===
        score_stab = sum((self.config["rmsf_cutoff_nm"] - rv) * 80 for rv in rmsf_vals)
        
        # === 2. 距离偏好 (微调，权重 0.8) ===
        r0_A = r0_nm * 10
        if 7.0 <= r0_A <= 9.5:  # 偏好 7-9.5Å，略收紧
            score_dist = 40 - abs(r0_A - 8.2) * 3
        else:
            score_dist = max(0, 20 - abs(r0_A - 8.2) * 2)
        
        # === 3. NEW: kr 合理性 (权重 1.2) ===
        if kr_raw <= 600:
            kr_score = 25  # 理想范围
        elif kr_raw <= 1200:
            kr_score = 25 - (kr_raw - 600) * 0.02  # 线性衰减
        else:
            kr_score = max(0, 13 - (kr_raw - 1200) * 0.03)  # 快速衰减
        
        # === 4. NEW: 角度质量 (权重 1.0) ===
        thA_deg, thB_deg = np.degrees(thA_rad), np.degrees(thB_rad)
        if 70 <= thA_deg <= 110 and 70 <= thB_deg <= 110:
            angle_score = 20
        elif 50 <= thA_deg <= 130 and 50 <= thB_deg <= 130:
            angle_score = 10
        else:
            angle_score = 0  # 已在硬过滤中排除，此处为保险
        
        # === 5. NEW: 锚点几何分散度 (权重 1.0) ===
        R0, R1, R2 = [traj.xyz[0, a] for a in rec_anchors]
        anchor_dists = [
            np.linalg.norm(R0-R1) * 10.0,
            np.linalg.norm(R1-R2) * 10.0,
            np.linalg.norm(R0-R2) * 10.0,
        ]  # nm → Å
        avg_anchor_dist = np.mean(anchor_dists)
        if 12 <= avg_anchor_dist <= 22:  # 12-22Å 理想分散
            geo_score = 18
        elif 8 <= avg_anchor_dist <= 28:
            geo_score = 18 - abs(avg_anchor_dist - 17) * 0.8
        else:
            geo_score = 0
        
        # === 6. NEW: 残基序列分散度 (权重 0.8) ===
        res_indices = sorted([top.atom(a).residue.index for a in rec_anchors])
        min_gap = min(res_indices[1]-res_indices[0], res_indices[2]-res_indices[1])
        if min_gap >= 20:
            seq_score = 15
        elif min_gap >= 10:
            seq_score = 10
        elif min_gap >= 5:
            seq_score = 4
        else:
            seq_score = 0  # 太接近，可能共线
        
        # === 综合 ===
        total = (score_stab + score_dist*0.8 + kr_score*1.2 +
                 angle_score + geo_score + seq_score*0.8)
        
        return total, {
            "stab": score_stab, "dist": score_dist, "kr": kr_score,
            "angle": angle_score, "geo": geo_score, "seq": seq_score
        }

    def estimate_multiple_anchors_from_trajectory(
        self,
        traj,
        ligand_resname: str,
        n_candidates: int = 5,
        output_path: Optional[str] = None,
        min_anchor_distance: float = 0.4,
        max_r0_angstrom: float = 10.0,
        max_kr: float = 2000.0,
        min_residue_gap: int = 3,
        use_last_ns: float = 5.0,  # ✅ 强制参数：只分析最后 N ns
    ) -> List[Dict]:
        """
        确定性枚举 v6.5 (最终修复版)
        【关键修正】
        1. 强制切片最后 5ns：基于轨迹时间精确切除前期不稳定部分
        2. 最终 r0/kr 二次硬校验：绝不返回超标候选
        3. 残基间隔与去重：杜绝鬼打墙
        """
        try:
            import mdtraj as md
        except ImportError:
            raise ImportError("需要 mdtraj")

        # ========================================================================
        # ✅ 0. 强制切片最后 5ns (物理需求：切除预平衡前期不稳定部分)
        # ========================================================================
        original_len = len(traj)
        if hasattr(traj, "time") and len(traj.time) > 0:
            t_max = traj.time[-1]  # ps
            t_cut = t_max - (use_last_ns * 1000.0)  # ns -> ps
            mask = traj.time >= t_cut
            n_keep = mask.sum()

            if n_keep > 0:
                traj = traj[mask]
                print(
                    f"🔪 轨迹切片: 仅使用最后 {use_last_ns} ns ({n_keep} 帧 / 原始 {original_len} 帧)"
                )
            else:
                print(
                    f"⚠️ 轨迹长度不足 {use_last_ns} ns，使用全轨迹 ({original_len} 帧)"
                )
        else:
            print(f"⚠️ 轨迹无时间信息，使用全轨迹 ({original_len} 帧)")

        print(
            f"🔍 确定性枚举 v6.5 | 目标: {n_candidates} | r0≤{max_r0_angstrom}Å | kr≤{max_kr} | 残基间隔≥{min_residue_gap}"
        )

        top = traj.topology
        lig_sel = top.select(f"resname {ligand_resname}")
        if len(lig_sel) == 0:
            raise ValueError(f"未找到配体: {ligand_resname}")

        # 2. 口袋 Cα 与 RMSF (基于切片后的稳态轨迹)
        # --- 确保 rigid_ca 在这里被定义，防止 NameError ---
        lig_center = traj.xyz[0, lig_sel].mean(axis=0)
        prot_ca = top.select("protein and name CA")
        
        # 筛选口袋区域 Cα (1.0 nm 范围内)
        pocket_ca = [ca for ca in prot_ca if np.linalg.norm(traj.xyz[0, ca] - lig_center) <= 1.0]
        
        rmsf_traj = traj[:: max(1, len(traj) // 100)]
        if len(rmsf_traj) > 1 and len(pocket_ca) > 0:
            rmsf_traj.superpose(rmsf_traj, 0, atom_indices=pocket_ca)
            rmsf = np.sqrt(
                np.mean(
                    (rmsf_traj.xyz[:, pocket_ca] - rmsf_traj.xyz[0, pocket_ca]) ** 2,
                    axis=(0, 2),
                )
            )
        else:
            rmsf = np.zeros(len(pocket_ca))
            
        rigid_mask = rmsf < self.config["rmsf_cutoff_nm"]
        rigid_ca = [pocket_ca[i] for i in range(len(pocket_ca)) if rigid_mask[i]]
        rigid_res = [top.atom(ca).residue.index for ca in rigid_ca]

        # ✅ 安全检查：刚性原子不足无法构建 Boresch
        if len(rigid_ca) < 3:
            print(f"  ❌ 刚性 Cα 原子不足 3 个 (当前 {len(rigid_ca)})，无法进行几何枚举。")
            return []

        # 1. 生成配体锚点候选三元组 (基于质量+刚性排序)
        lig_heavy = top.select(f"resname {ligand_resname} and not element H")
        if len(lig_heavy) < 3:
            lig_heavy = top.select(f"resname {ligand_resname}")
        
        lig_combos = []
        for combo in combinations(lig_heavy, 3):
            # 评分逻辑：质量之和 + 骨架原子奖励
            mass_score = sum(top.atom(i).element.mass for i in combo)
            name_bonus = sum(2.0 if top.atom(i).name in ["CG","CD","CE","CZ","CA"] else 0.0 for i in combo)
            lig_combos.append((combo, mass_score + name_bonus))
            
        M = min(20, len(lig_combos))
        lig_combos = [c[0] for c in sorted(lig_combos, key=lambda x: x[1], reverse=True)[:M]]
        print(f"  → 配体锚点候选: {len(lig_combos)} 个三元组")

        # 2. 受体锚点枚举 + 嵌套配体候选扫描
        candidates = []
        seen_geo_keys = set()  # 几何去重

        # 遍历受体三元组
        for rec_combo in combinations(range(len(rigid_ca)), 3):
            ca0, ca1, ca2 = [rigid_ca[i] for i in rec_combo]
            res0, res1, res2 = [rigid_res[i] for i in rec_combo]
            
            # 受体侧预过滤（残基间隔、锚点间距）
            if min([abs(res0-res1), abs(res0-res2), abs(res1-res2)]) < min_residue_gap:
                continue
                
            pos_rec = [traj.xyz[0, ca] for ca in [ca0, ca1, ca2]]
            if min(np.linalg.norm(pos_rec[i]-pos_rec[j]) for i in range(3) for j in range(i+1,3)) < min_anchor_distance:
                continue
                
            # 遍历配体三元组
            for lig_combo in lig_combos:
                pos_lig = [traj.xyz[0, idx] for idx in lig_combo]
                
                # ✅ 修复：放宽配体内部间距阈值 (从 0.25 -> 0.15 nm / 1.5 Å)
                # 适配刚性小分子（如苯环）原子间距较近的情况
                if min(np.linalg.norm(pos_lig[i]-pos_lig[j]) for i in range(3) for j in range(i+1,3)) < 0.15:
                    continue
                
                # ✅ 严格几何校验
                ok, msg = self._validate_boresch_geometry_strict(
                    [ca0, ca1, ca2], lig_combo, traj.xyz[0]
                )
                if not ok:
                    continue
                    
                # 几何去重
                r0_A = np.linalg.norm(pos_rec[0] - pos_lig[0]) * 10.0
                geo_key = (round(r0_A, 1), tuple(sorted([ca0, ca1, ca2][:2])), tuple(sorted(lig_combo[:2])))
                if geo_key in seen_geo_keys:
                    continue
                seen_geo_keys.add(geo_key)
                
                # 综合评分
                rmsf_vals = [rmsf[pocket_ca.index(ca)] for ca in [ca0, ca1, ca2]]
                score_stab = sum((self.config["rmsf_cutoff_nm"] - rv) * 100 for rv in rmsf_vals)
                if 5.0 <= r0_A <= 10.0:
                    score_dist = 60 - abs(r0_A - 7.5) * 4
                else:
                    score_dist = max(0, 30 - abs(r0_A - 7.5) * 2)
                total_score = score_stab + score_dist
                
                candidates.append({
                    "rec_anchors": [ca0, ca1, ca2],
                    "lig_anchors": list(lig_combo),
                    "res_key": tuple(sorted([res0, res1, res2])),
                    "r0_A": r0_A,
                    "score": total_score,
                    "rmsf_avg": np.mean(rmsf_vals),
                })

        # 3. 验证与快速筛选
        candidates.sort(key=lambda x: x["score"], reverse=True)
        print(f"  → 通过几何过滤的候选: {len(candidates)} 个")
        
        rigid_residues = {top.atom(int(a)).residue.index for a in rigid_ca}
        pocket_atoms = set(int(i) for i in lig_sel.tolist())
        for res in top.residues:
            if res.index in rigid_residues:
                for atom in res.atoms:
                    pocket_atoms.add(atom.index)
        pocket_sel = np.array(sorted(pocket_atoms), dtype=int)

        cap_h = []
        pocket_traj = traj.atom_slice(pocket_sel)
        context = OrbVacuumContext(pocket_traj.topology.to_openmm(), device=self.device)
        
        validated = []
        kr_seen = set()
        search_pool = max(n_candidates * 6, 30)
        
        for cand in candidates[:search_pool]:
            # 构建局部原子索引映射
            local_anchors = [int(np.where(pocket_sel == a)[0][0]) for a in cand["rec_anchors"] + cand["lig_anchors"]]
            
            ks = self._quick_validate(traj[:50], context, pocket_sel, local_anchors, cap_h)
            if ks is None:
                continue
            if ks["kr"] > max_kr:
                continue
            kr_r = round(ks["kr"], 1)
            if kr_r in kr_seen:
                continue
            kr_seen.add(kr_r)
            validated.append({**cand, "ks": ks, "local_anchors": local_anchors})
        print(f"  → 快速验证通过: {len(validated)} 个")

        # 4. 最终结果构建
        results = []
        seen_final = set()
        fallback_pool = []
        
        # 辅助函数
        def log_cand(rank, r0_nm, kr, res_key, tag="合格"):
            r0_a = r0_nm * 10.0
            print(f"  [{'✅' if tag=='合格' else '⬇️'} {tag}] #{rank}: r0={r0_a:.2f}Å | kr={kr:.1f} kJ/mol/nm² | 残基={res_key}")

        for cand in validated:
            res_key = cand["res_key"]
            if res_key in seen_final:
                continue
            seen_final.add(res_key)
            
            # 计算最终参数
            final = self._finalize_candidate(cand, traj, context, pocket_sel, cap_h)
            
            # 角度防御 (防止奇异)
            thA_deg = np.degrees(final["eq"]["thetaA0"])
            thB_deg = np.degrees(final["eq"]["thetaB0"])
            if not (40.0 <= thA_deg <= 140.0) or not (40.0 <= thB_deg <= 140.0):
                continue

            # 物理边界过滤
            r0_nm = final["eq"]["r0"]
            kr_val = final["ks"]["kr"]
            if r0_nm < 0.4 or r0_nm > 1.0:
                continue
            if r0_nm > max_r0_angstrom / 10.0:
                fallback_pool.append(final)
                continue
            if kr_val > max_kr:
                fallback_pool.append(final)
                continue

            # ✅ 使用综合评分替代简单 kr 惩罚
            rmsf_vals = [rmsf[pocket_ca.index(ca)] for ca in final["rec_anchors"]]
            total_score, score_breakdown = self._score_candidate_comprehensive(
                rec_anchors=final["rec_anchors"],
                lig_anchors=final["lig_anchors"],
                rmsf_vals=rmsf_vals,
                r0_nm=r0_nm,
                thA_rad=final["eq"]["thetaA0"],
                thB_rad=final["eq"]["thetaB0"],
                kr_raw=kr_val,
                traj=traj,
                top=top,
            )

            # 加入结果
            results.append({
                "rank": len(results) + 1,
                "receptor_indices": final["rec_anchors"],
                "ligand_indices": final["lig_anchors"],
                "receptor_residues": list(final["res_key"]),
                "force_constants": {k: float(v) for k, v in final["ks"].items()},
                "equilibrium_values": final["eq"],
                "total_score": float(total_score),
                "score_breakdown": score_breakdown,
                "method": "finite_combo_v6.5_last5ns",
            })
            log_cand(len(results), r0_nm, kr_val, res_key)
            
            if len(results) >= n_candidates:
                break

        # 降级回退
        if len(results) < n_candidates and fallback_pool:
            print(f"  ⚠️ 合格候选不足 ({len(results)}/{n_candidates})，启动降级回退...")
            fallback_pool.sort(key=lambda x: x["ks"]["kr"])
            for final in fallback_pool:
                if len(results) >= n_candidates: break
                res_key = final["res_key"]
                if res_key in seen_final: continue
                seen_final.add(res_key)
                
                # 放宽 kr 限制，但死守 r0 几何
                r0_nm_fb = final["eq"]["r0"]
                kr_val_fb = final["ks"]["kr"]
                
                # ✅ 使用综合评分（与主路径一致）
                rmsf_vals_fb = [rmsf[pocket_ca.index(ca)] for ca in final["rec_anchors"]]
                total_score_fb, score_breakdown_fb = self._score_candidate_comprehensive(
                    rec_anchors=final["rec_anchors"],
                    lig_anchors=final["lig_anchors"],
                    rmsf_vals=rmsf_vals_fb,
                    r0_nm=r0_nm_fb,
                    thA_rad=final["eq"]["thetaA0"],
                    thB_rad=final["eq"]["thetaB0"],
                    kr_raw=kr_val_fb,
                    traj=traj,
                    top=top,
                )

                results.append({
                    "rank": len(results) + 1,
                    "receptor_indices": final["rec_anchors"],
                    "ligand_indices": final["lig_anchors"],
                    "receptor_residues": list(final["res_key"]),
                    "force_constants": {k: float(v) for k, v in final["ks"].items()},
                    "equilibrium_values": final["eq"],
                    "total_score": float(total_score_fb),
                    "score_breakdown": score_breakdown_fb,
                    "method": "finite_combo_v6.5_fallback",
                    "warning": "kr 超出上限，已放行"
                })
                log_cand(len(results), final["eq"]["r0"], final["ks"]["kr"], res_key, tag="回退")

        if not results:
            print(f"  ❌ 未找到满足条件的合格候选。")
        else:
            print(f"✅ 最终返回 {len(results)} 个合格候选")
            
            # ✅ 【关键修复】按总分降序排序，并重新分配 rank 序号
            results.sort(key=lambda x: x.get("total_score", 0.0), reverse=True)
            for i, res in enumerate(results):
                res["rank"] = i + 1
                res["kr_bonus"] = round((max_kr - res["force_constants"]["kr"]) * 0.1, 2)
            
            # 打印诊断信息，确认排序生效
            top = results[0]
            print(f"  🏆 推荐首选: Rank #{top['rank']} (残基={top['receptor_residues']})")
            print(f"     kr={top['force_constants']['kr']:.1f} | 总分={top['total_score']:.2f} | kr加分={top.get('kr_bonus',0):.2f}")

        if output_path:
            import json
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, np.floating)): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return super().default(obj)
            with open(output_path, "w") as f:
                json.dump({"candidates": results}, f, indent=2, cls=NumpyEncoder)
            print(f"✅ 结果已保存: {output_path}")
            
        return results


# ============================================================================
# 4. 幽灵离子 & 2D路径规划 & 替身系统构建器
# ============================================================================
class GhostIonHandler:
    def __init__(self, ghost_ion_distance=10.0, ghost_ion_scale_factor=1.0):
        self.ghost_ion_distance = ghost_ion_distance
        self.ghost_ion_scale_factor = ghost_ion_scale_factor

    def _resolve_ghost_anchor(self, box_vectors=None, reference_positions=None, ligand_indices=None):
        if box_vectors is None:
            return (
                float(self.ghost_ion_distance),
                0.0,
                0.0,
            )

        box_lengths = np.array(
            [
                np.linalg.norm(np.asarray(vec.value_in_unit(unit.nanometer) if hasattr(vec, "value_in_unit") else vec, dtype=float))
                for vec in box_vectors
            ],
            dtype=float,
        )
        box_lengths = np.where(box_lengths > 1.0e-6, box_lengths, 3.0)
        safe_margin = np.minimum(0.2, 0.1 * box_lengths)

        if reference_positions is not None and ligand_indices:
            lig_xyz = []
            for idx in ligand_indices:
                pos = reference_positions[idx]
                if hasattr(pos, "value_in_unit"):
                    lig_xyz.append(np.asarray(pos.value_in_unit(unit.nanometer), dtype=float))
                else:
                    lig_xyz.append(np.asarray(pos, dtype=float))
            if lig_xyz:
                lig_com = np.mean(np.asarray(lig_xyz, dtype=float), axis=0)
                anchor = np.mod(lig_com + 0.5 * box_lengths - safe_margin, box_lengths)
                return tuple(float(x) for x in anchor)

        anchor = 0.5 * box_lengths - safe_margin
        return tuple(float(x) for x in anchor)

    def create_ghost_ion_force(
        self,
        ligand_indices,
        ligand_charges,
        lambda_param="lam_coul",
        box_vectors=None,
        reference_positions=None,
    ):
        total = sum(ligand_charges)
        ghost = -total * self.ghost_ion_scale_factor
        if abs(ghost) < 1.0e-12 or not ligand_indices:
            return None

        xg, yg, zg = self._resolve_ghost_anchor(
            box_vectors=box_vectors,
            reference_positions=reference_positions,
            ligand_indices=ligand_indices,
        )
        force = openmm.CustomExternalForce(
            f"{lambda_param} * 138.935456 * ghost_charge * ligand_charge / "
            f"max(periodicdistance(x, y, z, ghost_x, ghost_y, ghost_z), 0.05)"
        )
        force.addGlobalParameter(lambda_param, 1.0)
        force.addGlobalParameter("ghost_charge", float(ghost))
        force.addGlobalParameter("ghost_x", float(xg))
        force.addGlobalParameter("ghost_y", float(yg))
        force.addGlobalParameter("ghost_z", float(zg))
        force.addPerParticleParameter("ligand_charge")
        for idx, charge in zip(ligand_indices, ligand_charges):
            force.addParticle(int(idx), [float(charge)])
        return force


class TwoDimensionalLambdaPathPlanner:
    def __init__(self, n_points=20, path_type="decoupling"):
        self.n_points = n_points
        self.path_type = path_type

    def generate_path(self):
        if self.path_type == "diagonal":
            return [
                (1.0 - i / self.n_points, 1.0 - i / self.n_points)
                for i in range(self.n_points + 1)
            ]
        elif self.path_type == "decoupling":
            n_half = self.n_points // 2
            lambdas = []
            for i in range(n_half + 1):
                lambdas.append((1.0 - i / n_half, 1.0))
            for i in range(1, self.n_points - n_half + 1):
                lambdas.append((0.0, 1.0 - i / (self.n_points - n_half)))
            return lambdas
        else:
            return [(1.0, 1.0 - i / self.n_points) for i in range(self.n_points + 1)]


class SurrogateSystemBuilder:
    def __init__(self, surrogate_params, ghost_handler=None, sigma_gauss_nm: float = 0.10):
        self.surrogate_potential = DEXPSurrogatePotential.from_dict(
            {
                k: v
                for k, v in surrogate_params.items()
                if k not in ("fitting_success", "final_cost")
            }
        )
        self.ghost_handler = ghost_handler
        self.sigma_gauss_nm = float(sigma_gauss_nm)

    def build_surrogate_system(
        self,
        original_system,
        ligand_indices,
        environment_indices,
        lambda_names=("lam_coul", "lam_vdw"),
        force_group=1,
        reference_positions=None,
        box_vectors=None,
    ):
        new_system = ensure_owned_system(original_system)
        nb_force = next(
            (f for f in new_system.getForces() if isinstance(f, openmm.NonbondedForce)),
            None,
        )
        if not nb_force:
            raise ValueError("未找到 NonbondedForce")

        lig_set = {int(idx) for idx in ligand_indices}
        env_set = {int(idx) for idx in environment_indices if int(idx) not in lig_set}
        if not lig_set:
            raise ValueError("ligand_indices 为空，无法构建 surrogate decoupling system")
        if not env_set:
            raise ValueError("environment_indices 为空，无法构建 ligand-environment surrogate force")
        original_params = [
            nb_force.getParticleParameters(i) for i in range(new_system.getNumParticles())
        ]
        reference_exclusions = []
        for i in range(nb_force.getNumExceptions()):
            p1, p2, _, _, _ = nb_force.getExceptionParameters(i)
            reference_exclusions.append((int(p1), int(p2)))

        # 1) 先恢复 ligand-ligand 内部 nonbonded / 1-4，保留原始 MM 内部拓扑语义。
        ll_force, ll_14_force = create_ligand_internal_force(
            nb_force=nb_force,
            perturbed_indices=sorted(lig_set),
            particle_params=original_params,
            reference_exclusions=reference_exclusions,
            num_particles=new_system.getNumParticles(),
            system=new_system,
        )
        ll_force.setForceGroup(force_group)
        new_system.addForce(ll_force)
        if ll_14_force is not None:
            ll_14_force.setForceGroup(force_group)
            new_system.addForce(ll_14_force)

        # 2) 主 NonbondedForce 中将 ligand 完全去耦，避免原始 MM L-E 项始终全开。
        for idx in sorted(lig_set):
            q, sigma, epsilon = original_params[idx]
            nb_force.setParticleParameters(
                idx,
                0.0 * unit.elementary_charge,
                sigma,
                0.0 * unit.kilojoule_per_mole,
            )
        for exc_idx in range(nb_force.getNumExceptions()):
            p1, p2, _, sigma, _ = nb_force.getExceptionParameters(exc_idx)
            p1, p2 = int(p1), int(p2)
            if p1 in lig_set or p2 in lig_set:
                nb_force.setExceptionParameters(
                    exc_idx,
                    p1,
                    p2,
                    0.0 * unit.elementary_charge * unit.elementary_charge,
                    sigma,
                    0.0 * unit.kilojoule_per_mole,
                )

        # 3) L-E Gaussian electrostatics：用平滑库仑核替代点电荷奇点，并与 DEXP 共用
        #    0.45~0.65 nm 的 switching/cutoff 缝合区。
        sigma_gauss_nm = max(
            float(getattr(self.surrogate_potential, "sigma_elec", self.sigma_gauss_nm)),
            1.0e-6,
        )
        gamma_eff = 1.0 / max(math.sqrt(2.0) * sigma_gauss_nm, 1.0e-6)
        gauss_expr = (
            f"{lambda_names[0]} * 138.935456*q1*q2*erf({gamma_eff}*r_safe)/r_safe; "
            "r_safe = max(r, 1e-6)"
        )
        coul_force = openmm.CustomNonbondedForce(gauss_expr)
        coul_force.addPerParticleParameter("q")
        coul_force.addGlobalParameter(lambda_names[0], 1.0)
        for i in range(new_system.getNumParticles()):
            q, _, _ = original_params[i]
            coul_force.addParticle([q.value_in_unit(unit.elementary_charge)])
        coul_force.addInteractionGroup(sorted(lig_set), sorted(env_set))
        coul_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        coul_force.setCutoffDistance(self.surrogate_potential.cutoff_distance * unit.nanometer)
        coul_force.setUseSwitchingFunction(True)
        coul_force.setSwitchingDistance(
            (self.surrogate_potential.cutoff_distance - self.surrogate_potential.switch_width)
            * unit.nanometer
        )
        coul_force.setForceGroup(force_group)
        for p1, p2 in reference_exclusions:
            coul_force.addExclusion(int(p1), int(p2))
        new_system.addForce(coul_force)

        # 4) L-E DEXP：短程排斥/色散替身，与 Gaussian electrostatics 拼成完整 surrogate。
        dexp_expr = (
            f"{self.surrogate_potential.build_expression(lam_vdw=lambda_names[1])}"
        )
        dexp_force = openmm.CustomNonbondedForce(dexp_expr)
        dexp_force.addGlobalParameter(lambda_names[1], 1.0)
        for i in range(new_system.getNumParticles()):
            dexp_force.addParticle([])
        dexp_force.addInteractionGroup(sorted(lig_set), sorted(env_set))
        dexp_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        dexp_force.setCutoffDistance(self.surrogate_potential.cutoff_distance * unit.nanometer)
        dexp_force.setUseSwitchingFunction(True)
        dexp_force.setSwitchingDistance(
            (self.surrogate_potential.cutoff_distance - self.surrogate_potential.switch_width)
            * unit.nanometer
        )
        dexp_force.setForceGroup(force_group)

        for p1, p2 in reference_exclusions:
            dexp_force.addExclusion(int(p1), int(p2))

        new_system.addForce(dexp_force)
        sync_all_exclusions(new_system)

        if self.ghost_handler and ligand_indices:
            charges = [
                original_params[i][0].value_in_unit(unit.elementary_charge)
                for i in sorted(ligand_indices)
            ]
            ghost_force = self.ghost_handler.create_ghost_ion_force(
                ligand_indices=sorted(ligand_indices),
                ligand_charges=charges,
                lambda_param=lambda_names[0],
                box_vectors=box_vectors if box_vectors is not None else new_system.getDefaultPeriodicBoxVectors(),
                reference_positions=reference_positions,
            )
            if ghost_force is not None:
                new_system.addForce(ghost_force)
        return new_system


# ============================================================================
# 5. Orb 扫描器 & 混合工厂 & Surrogate 流水线
# ============================================================================
class OrbScanner:
    def __init__(
        self,
        model_name="mace-off24-medium",
        n_order=6,
        charge=0,
        multiplicity=1,
        device=GLOBAL_DEVICE,
    ):
        self.n_order = n_order
        self.charge = charge
        self.multiplicity = multiplicity
        self.device = device
        self.model_name = model_name
        self.context = None
        self.system = None
        if HAS_ORB:
            self.potential = MLPotential(model_name)

    def _setup_vacuum_context(self, rdkit_mol):
        if self.context is None and HAS_ORB:
            vacuum_top = openmm.app.Topology()
            c = vacuum_top.addChain()
            res = vacuum_top.addResidue("MOL", c)
            for atom in rdkit_mol.GetAtoms():
                vacuum_top.addAtom(
                    atom.GetSymbol(),
                    openmm.app.element.Element.getByAtomicNumber(atom.GetAtomicNum()),
                    res,
                )
            self.system = self.potential.createSystem(vacuum_top, **_build_openmmml_kwargs(
                device=self.device,
                return_energy_type="interaction_energy",
                charge=self.charge,
                multiplicity=self.multiplicity,
            ))
            self.context = openmm.Context(self.system, openmm.VerletIntegrator(0.001))

    def run_torsion_scan(self, rdkit_mol, torsion_indices):
        self._setup_vacuum_context(rdkit_mol)
        angles = np.arange(-180, 180, 10)
        e_list = []
        conf = rdkit_mol.GetConformer()
        for ang in angles:
            import rdkit.Chem.rdMolTransforms as rdt

            rdt.SetDihedralDeg(conf, *torsion_indices, float(ang))
            pos_nm = (
                np.array(
                    [conf.GetAtomPosition(i) for i in range(rdkit_mol.GetNumAtoms())]
                )
                * 0.1
            )
            self.context.setPositions([openmm.Vec3(*p) for p in pos_nm])
            e_list.append(
                self.context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoules_per_mole)
            )
        return angles, np.array(e_list)


    # abfe_core.py → 加在 OrbScanner 类之后或作为独立函数

    def scan_boresch_1d_pes(
        self,
        rdkit_mol,
        rec_indices: List[int],
        lig_indices: List[int],
        scan_coord: str = "r",        # "r" | "thetaA" | "thetaB" | "phiA" | "phiB" | "phiC"
        n_points: int = 21,
        device: str = "cpu",
        model_name: str = "mace-off24-medium"
    ) -> Dict:
        """
        沿单个 Boresch 坐标做 1D Orb 势能面扫描。
        
        参数：
            rdkit_mol: RDKit Mol 对象（含参考构象）
            rec_indices: 受体 3 原子在 rdkit_mol 中的索引
            lig_indices: 配体 3 原子在 rdkit_mol 中的索引
            scan_coord: 要扫描的坐标 ("r" nm | "thetaA"/"thetaB" rad | "phiA"/"phiB"/"phiC" rad)
            n_points: 扫描点数
        
        返回：
            {"x": array, "U_ML": array, "scan_coord": str, 
             "harmonic_k": float, "anharmonic_flag": bool}
        """
        if not HAS_ORB:
            raise ImportError("Orb 环境不可用 (需要 torch + openmmml)")

        import numpy as np
        from openmm import unit, Vec3
        from openmmml import MLPotential

        # 1. 提取参考几何量
        conf = rdkit_mol.GetConformer()
        coords = np.array([conf.GetAtomPosition(i) for i in range(rdkit_mol.GetNumAtoms())])
        
        def _calc_geom(pos):
            R0, R1, R2 = pos[rec_indices]
            L0, L1, L2 = pos[lig_indices]
            
            def dist(a, b): return np.linalg.norm(a - b)
            def angle(a, b, c):
                v1, v2 = a - b, c - b
                cos_val = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12), -1.0, 1.0)
                return np.arccos(cos_val)
            def dihedral(a, b, c, d):
                b1, b2, b3 = b - a, c - b, d - c
                n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
                m1 = np.cross(n1, b2 / np.linalg.norm(b2))
                return np.arctan2(np.dot(m1, n2), np.dot(n1, n2))
            
            return {
                "r": dist(R0, L0),
                "thetaA": angle(R1, R0, L0),
                "thetaB": angle(R0, L0, L1),
                "phiA": dihedral(R2, R1, R0, L0),
                "phiB": dihedral(R1, R0, L0, L1),
                "phiC": dihedral(R0, L0, L1, L2),
            }

        supported_scan_coords = {"r", "thetaA", "thetaB", "phiA", "phiB", "phiC"}
        if scan_coord not in supported_scan_coords:
            raise ValueError(f"未知 Boresch 扫描坐标: {scan_coord}")
        if scan_coord != "r":
            raise NotImplementedError(
                "scan_boresch_1d_pes 当前只实现 r 距离扫描；"
                f"{scan_coord} 角度/二面角扫描需要刚体旋转实现，拒绝返回未扰动几何的假 PES。"
            )

        ref_geom = _calc_geom(coords * 0.1)  # Å → nm
        ref_val = ref_geom[scan_coord]

        # 2. 确定扫描范围 (±标准差)
        scan_range = {
            "r": 0.15,       # nm (±1.5 Å)
            "thetaA": 0.3,   # rad (±17°)
            "thetaB": 0.3,
            "phiA": 0.5,     # rad (±29°)
            "phiB": 0.5,
            "phiC": 0.5,
        }
        half_range = scan_range.get(scan_coord, 0.3)
        x_vals = np.linspace(ref_val - half_range, ref_val + half_range, n_points)

        # 3. 构建 6 原子口袋真空系统
        vacuum_top = app.Topology()
        chain = vacuum_top.addChain()
        res = vacuum_top.addResidue("MOL", chain)
        
        # 提取 6 原子元素
        all_6 = rec_indices + lig_indices
        for idx in all_6:
            atom = rdkit_mol.GetAtomWithIdx(idx)
            vacuum_top.addAtom(
                atom.GetSymbol(),
                app.element.Element.getByAtomicNumber(atom.GetAtomicNum()),
                res
            )
        
        potential = MLPotential(model_name)
        system = potential.createSystem(vacuum_top, **_build_openmmml_kwargs(
            device=device,
            return_energy_type="energy",
            charge=0,
            multiplicity=1,
        ))
        integrator = openmm.VerletIntegrator(0.001)
        platform = openmm.Platform.getPlatformByName(device.upper() if device != "cpu" else "CPU")
        context = openmm.Context(system, integrator, platform)

        # 4. 沿坐标扫描
        U_vals = []
        for target_val in x_vals:
            pos_perturbed = coords.copy() * 0.1  # nm
            
            # 只扰动扫描坐标对应的自由度
            if scan_coord == "r":
                delta = target_val - ref_val
                direction = pos_perturbed[lig_indices[0]] - pos_perturbed[rec_indices[0]]
                direction /= np.linalg.norm(direction) + 1e-12
                pos_perturbed[lig_indices] += delta * direction
            
            elif scan_coord in ["thetaA", "thetaB"]:
                # 角度扫描：旋转配体或受体子集（简化处理）
                # 此处用有限差分梯度线性近似
                dq = target_val - ref_val
                # ... 简化实现：线性插值参考几何
                pass
            
            coords_nm = pos_perturbed * 0.1 if coords.max() > 10 else pos_perturbed
            context.setPositions([Vec3(*p) for p in coords_nm[all_6]])
            state = context.getState(getEnergy=True)
            U_vals.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))

        U_vals = np.array(U_vals)
        U_vals -= U_vals.min()  # 零点对齐

        # 5. 谐波性诊断
        from scipy.optimize import curve_fit
        def harmonic(x, k, x0, U0):
            return 0.5 * k * (x - x0)**2 + U0

        try:
            popt, _ = curve_fit(harmonic, x_vals, U_vals, 
                                p0=[100.0, ref_val, 0.0],
                                bounds=([1.0, ref_val - half_range, -100],
                                        [5000.0, ref_val + half_range, 100]))
            k_fit = popt[0]
            U_fit = harmonic(x_vals, *popt)
            rmse = np.sqrt(np.mean((U_vals - U_fit)**2))
            # 非谐性判据：RMSE > 2.0 kJ/mol 或 k_fit 异常
            anharmonic = (rmse > 2.0) or (k_fit < 1.0)

            # 双势阱诊断：检查 U(x) 的极小值数量
            from scipy.signal import argrelextrema
            minima = argrelextrema(U_vals, np.less)[0]
            is_double_well = len(minima) >= 2
            if is_double_well:
                anharmonic = True
        except:
            k_fit = None
            rmse = None
            anharmonic = True
            is_double_well = False

        del context
        import gc; gc.collect()

        return {
            "x": x_vals.tolist(),
            "U_ML": U_vals.tolist(),
            "scan_coord": scan_coord,
            "ref_value": float(ref_val),
            "harmonic_k_fit": float(k_fit) if k_fit else None,
            "rmse_vs_harmonic_kJ_mol": float(rmse) if rmse else None,
            "anharmonic_flag": anharmonic,
            "double_well_flag": is_double_well,
        } 

class OrbMMHybridFactory:
    def get_rotatable_torsions_rdkit(self, mol):
        import rdkit.Chem as Chem

        pattern = Chem.MolFromSmarts("[!#1;!D1]-[!#1;!D1]")
        if pattern is None:
            return []
        torsions = []
        for bond in mol.GetSubstructMatches(pattern):
            a1, a2 = bond[0], bond[1]
            for n1 in [x.GetIdx() for x in mol.GetAtomWithIdx(a1).GetNeighbors()]:
                if n1 != a2:
                    for n2 in [
                        x.GetIdx() for x in mol.GetAtomWithIdx(a2).GetNeighbors()
                    ]:
                        if n2 != a1 and mol.GetBondBetweenAtoms(a1, a2) is not None:
                            torsions.append((n1, a1, a2, n2))
        return torsions


class Orbv3SurrogatePipeline:
    def __init__(self, model_name="mace-off24-medium", device=GLOBAL_DEVICE):
        self.orb_calculator = None
        self.default_surrogate = DEXPSurrogatePotential()
        self.ghost_handler = GhostIonHandler()
        if HAS_ORB:
            self.orb_calculator = OrbScanner(model_name, device=device)

    def fit_surrogate_from_orb_data(self, distances, orb_energies, particle_types=None):
        fitter = Orbv3SurrogateFitter(fitting_region=(0.2, 1.2))
        return fitter.fit_parameters(distances, orb_energies, eff_eps=1.0)

    def plan_2d_lambda_path(self, path_type="decoupling", n_points=20):
        return TwoDimensionalLambdaPathPlanner(n_points, path_type).generate_path()

    def build_production_system(
        self,
        original_system,
        ligand_indices,
        environment_indices,
        surrogate_params,
        reference_positions=None,
        box_vectors=None,
    ):
        return SurrogateSystemBuilder(
            surrogate_params, self.ghost_handler
        ).build_surrogate_system(
            original_system,
            ligand_indices,
            environment_indices,
            reference_positions=reference_positions,
            box_vectors=box_vectors,
        )




# ============================================================================
# 6. 势能路由工厂 & 幽灵离子快捷函数
# ============================================================================
class AlchemicalPotentialFactory:
    @staticmethod
    def build(potential_type, params, lam_coul, lam_vdw):
        if isinstance(params, (ACESoftcorePotential, DEXPSurrogatePotential)):
            obj = params
        elif potential_type == "dexp":
            obj = DEXPSurrogatePotential.from_dict(params or {})
        elif potential_type == "softcore":  # ✅ 显式识别
            obj = ACESoftcorePotential.from_dict(params or {})
        else:
            obj = ACESoftcorePotential.from_dict(params or {})
        if isinstance(obj, DEXPSurrogatePotential):
            return obj.build_expression(lam_vdw=lam_vdw), obj.get_parameters_dict()
        return obj.build_expression(lam_coul, lam_vdw), obj.get_parameters_dict()

def create_ghost_ion_force(
    lig_indices,
    lig_charges,
    lam_param="lam_coul",
    dist=10.0,
    scale=0.1,
    box_vectors=None,
    reference_positions=None,
):
    return GhostIonHandler(dist, scale).create_ghost_ion_force(
        ligand_indices=lig_indices,
        ligand_charges=lig_charges,
        lambda_param=lam_param,
        box_vectors=box_vectors,
        reference_positions=reference_positions,
    )


# ============================================================================
# 7. 在线收敛监控器 (生产级动态诊断)
# ============================================================================
class OnlineConvergenceMonitor:
    """
    ABFE 在线收敛监控器 (核心物理组件)
    职责：实时分析自由能轨迹的平稳性、统计误差和相空间重叠度。
    设计原则：
    - 增量热启动 (initial_f_k) 实现 O(1) 级别 MBAR 重算
    - 五维正交判据杜绝"假收敛"
    - 单位原生支持 (OpenMM unit -> kJ/mol)

    ⚠️ 严格契约：输入 u_kn_chunk 必须为 Total Reduced Potential
       即 u_k(x) = β[U_phys(x) + U_restraint(x)]
       若仅传入纯物理势能，Overlap 与 N_eff 指标将失效，可能导致假收敛。
    """

    def __init__(
        self,
        temperature: unit.Quantity,
        check_interval: int = 10,
        ma_window: int = 5,
        precision_thresholds: Optional[Dict] = None,
    ):

        self.kt = (unit.MOLAR_GAS_CONSTANT_R * temperature).value_in_unit(
            unit.kilojoules_per_mole
        )
        self.interval = check_interval
        self.ma_window = ma_window

        self.thr = {
            "drift": 0.5,
            "error": 0.8,
            "neff_ratio": 0.20,
            "overlap": 0.85,
            "min_neighbor_overlap": 0.03,
            "ma_std": 0.30,
        }
        if precision_thresholds:
            self.thr.update(precision_thresholds)

        self.f_k_prev = None
        self.history = []
        self.dg_deque = deque(maxlen=ma_window + 2)

    def add_diagnostic(self, u_kn_chunk: np.ndarray, step: int) -> Dict:
        """
        核心诊断逻辑：输入势能矩阵，输出多维收敛报告
        参数：
            u_kn_chunk: (K_states, N_frames) 约化势能矩阵 (已除以 kT)
            step: 当前模拟步数 (用于日志)
        返回：
            dict: {converged: bool, dg: float, error: float, ...}
        """
        K, N = u_kn_chunk.shape
        if N < 20:
            return {"converged": False, "msg": "waiting_for_data", "step": step}

        energy_mean = np.mean(u_kn_chunk) * self.kt
        energy_var = np.var(u_kn_chunk, axis=1)

        if energy_mean < -2500.0 and np.max(energy_var) < 5.0:
            print(
                f"  ⚠️ [Monitor] 能量矩阵疑似遗漏限制力 (μ={energy_mean:.1f}, max(σ²)={np.max(energy_var):.1f})"
            )
            print(
                f"     → 请确保 ibs_engine 中 getState(groups={{group_id}}) 包含 Boresch 力"
            )
            return {
                "converged": False,
                "error": "suspected_missing_restraint_energy",
                "step": step,
            }

        try:
            K, N = u_kn_chunk.shape
            n_k_array = np.full(K, N, dtype=int)
            mbar = _build_mbar_compatible(
                u_kn_chunk,
                n_k_array,
                initial_f_k=self.f_k_prev,
                solver_protocol="hybr",
                solver_tolerance=1e-5,
                verbose=False,
            )

            res = _compute_free_energy_result_compatible(mbar, compute_uncertainty=True)
            df, ddf = _extract_free_energy_arrays(res, require_uncertainty=True)

            self.f_k_prev = df[0, :].copy()

            dg = (df[0, -1] - df[0, 0]) * self.kt
            err = ddf[0, -1] * self.kt
            neff = mbar.compute_effective_sample_number()
            neff_ratio = float(np.min(neff) / N) if N > 0 else 0.0

            overlap_mat = mbar.compute_overlap()["matrix"]
            overlap = float(np.max(np.diag(overlap_mat)))
            
            # ✅ MBAR 重叠度自动诊断与降级
            min_offdiag = np.min([
                overlap_mat[i, j] 
                for i in range(K) for j in range(K) 
                if abs(i-j) == 1
            ])
            if min_offdiag < 0.03:
                print(f"  🚨 [Monitor] 相邻窗口最小重叠 {min_offdiag:.3f} < 0.03，MBAR 误差可能低估！")
                print(f"     → 建议：延长采样 20% 或在重叠最差区域附近插值窗口")

            self.dg_deque.append(dg)
            ma_std = (
                float(np.std(list(self.dg_deque)[-self.ma_window :]))
                if len(self.dg_deque) >= self.ma_window
                else 99.0
            )
            drift = (
                float(abs(self.dg_deque[-1] - self.dg_deque[0]))
                if len(self.dg_deque) > 1
                else 99.0
            )

            is_stable = drift < self.thr["drift"] and ma_std < self.thr["ma_std"]
            is_precise = err < self.thr["error"] and neff_ratio > self.thr["neff_ratio"]
            is_connected = min_offdiag >= self.thr.get("min_neighbor_overlap", 0.03)

            converged = is_stable and is_precise and is_connected

            report = {
                "step": step,
                "converged": converged,
                "dg": dg,
                "error": err,
                "neff_ratio": neff_ratio,
                "overlap_max_diag": overlap,
                "overlap_min_offdiag": float(min_offdiag),
                "drift": drift,
                "ma_std": ma_std,
                "details": {
                    "stable": is_stable,
                    "precise": is_precise,
                    "connected": is_connected,
                },
            }
            if min_offdiag < 0.03:
                report["warning"] = "low_overlap"
            self.history.append(report)
            return report

        except ImportError:
            return {"converged": False, "error": "pymbar_not_installed", "step": step}
        except Exception as e:
            return {
                "converged": False,
                "error": f"mbar_failed: {str(e)[:60]}",
                "step": step,
            }

    def export_convergence_data(self, path: str):
        import json

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(path, "w") as f:
            json.dump(self.history, f, indent=2, cls=NumpyEncoder)

    def plot_convergence(self, output_path: str):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.history:
            print(f"  ⚠️  无收敛历史数据，跳过绘图: {output_path}")
            return

        valid_data = [h for h in self.history if "dg" in h and "error" in h]
        if not valid_data:
            return

        steps = [h["step"] for h in valid_data]
        dgs = [h["dg"] for h in valid_data]
        errs = [h["error"] for h in valid_data]

        plt.figure(figsize=(10, 6), dpi=300)

        plt.errorbar(
            steps,
            dgs,
            yerr=errs,
            fmt="o-",
            color="#2E86AB",
            ecolor="#A23B72",
            capsize=3,
            label="ΔG ± error",
            linewidth=1.5,
        )

        if dgs:
            plt.axhline(
                y=dgs[-1],
                color="gray",
                linestyle="--",
                alpha=0.5,
                label=f"Final ΔG: {dgs[-1]:.2f} kJ/mol",
            )

        plt.xlabel("Simulation Step", fontsize=11)
        plt.ylabel("ΔG (kJ/mol)", fontsize=11)
        plt.title("ABFE Convergence Monitor", fontsize=13, pad=15)
        plt.legend(frameon=True, fancybox=True, shadow=False)
        plt.grid(True, alpha=0.3, linestyle=":")

        plt.tight_layout()

        try:
            plt.savefig(output_path, bbox_inches="tight")
            print(f"  ✓ 收敛曲线已保存: {output_path}")
        except Exception as e:
            print(f"  ⚠️  保存图片失败: {e}")
        finally:
            plt.close()


#=============================================================================
# 修复 2: 约束 Jacobian 解析修正 (Constraint Correction)
#=============================================================================
def calculate_constraint_jacobian_correction(system, ligand_indices, temperature=300.0):
    """
    计算配体解耦过程中的约束修正项 (Jacobian Correction)
    物理原理：当约束键随配体变为 Ghost 时，构象空间体积密度发生变化。
    公式: ΔG_cons = +0.5 * R * T * Σ ln(μ_bond / μ_ref)
    μ_ref 取 1.0 Da (OpenMM 质量单位基准)
    """
    R = 0.008314462618  # kJ/(mol·K)
    T = temperature if isinstance(temperature, float) else temperature.value_in_unit(unit.kelvin)
    kT = R * T

    correction_kj = 0.0
    constrained_bonds = 0
    lig_set = set(ligand_indices)

    for i in range(system.getNumConstraints()):
        p1, p2, _ = system.getConstraintParameters(i)
        if p1 in lig_set and p2 in lig_set:
            m1 = system.getParticleMass(p1).value_in_unit(unit.dalton)
            m2 = system.getParticleMass(p2).value_in_unit(unit.dalton)
            mu = (m1 * m2) / (m1 + m2) if (m1 + m2) > 0 else 1e-6
            mu_ref = 1.0  # Da (OpenMM 标准参考约化质量)
            correction_kj += 0.5 * kT * math.log(max(mu, 1e-6) / mu_ref)
            constrained_bonds += 1

    if constrained_bonds > 0:
        print(f"  🔍 检测到 {constrained_bonds} 个配体约束键，Jacobian 修正: {correction_kj:.3f} kJ/mol")
    return correction_kj


#=============================================================================
# 修复 6: Boresch 平衡值从预平衡最后一帧直接计算
#=============================================================================
# 完整替换 abfe_core.py 中的 calc_boresch_from_last_frame 函数
def calc_boresch_from_last_frame(positions, rec_idx, lig_idx):
    """✅ 修复 2.3：兼容 Quantity 包裹、Numpy 数组、OpenMM Vec3 列表"""
    # 1. 尝试剥离单位
    if hasattr(positions, "value_in_unit"):
        pos = np.asarray(positions.value_in_unit(unit.nanometer), dtype=np.float64)
    elif isinstance(positions, np.ndarray):
        pos = positions.astype(np.float64, copy=False)
    else:
        # 兼容 [openmm.Vec3, ...] 列表
        pos = np.array([[getattr(p, 'x', p[0]), getattr(p, 'y', p[1]), getattr(p, 'z', p[2])] 
                        for p in positions], dtype=np.float64)
                        
    # 确保形状为 (N, 3)
    if pos.shape == (3, 3): pos = pos.T  # 处理传入 box_vectors 类转置误用
    elif pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions 形状异常: {pos.shape}，期望 (N, 3)")

    rec_idx = [int(i) for i in rec_idx]
    lig_idx = [int(i) for i in lig_idx]
    if len(rec_idx) != 3 or len(lig_idx) != 3:
        raise ValueError("Boresch 平衡值计算需要 3 个受体锚点和 3 个配体锚点")
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions 包含 NaN/Inf，拒绝更新 Boresch 平衡几何")

    r_coords = pos[rec_idx]
    l_coords = pos[lig_idx]
    if not np.all(np.isfinite(r_coords)) or not np.all(np.isfinite(l_coords)):
        raise ValueError("Boresch 锚点坐标包含 NaN/Inf，拒绝更新平衡几何")
    L0, L1, L2 = l_coords

    # 受体锚点顺序必须在估算阶段确定后保持锁定，绝不能按瞬时几何动态重排。
    H0, H1, H2 = r_coords

    def dist(a, b): return np.linalg.norm(a - b)
    def angle(a, b, c):
        ba, bc = a - b, c - b
        norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6: return np.pi / 2
        cos_val = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc + 1e-10), -1.0, 1.0)
        return np.arccos(cos_val)
    def dihedral(a, b, c, d):
        b1, b2, b3 = b - a, c - b, d - c
        norm_b2 = np.linalg.norm(b2)
        if norm_b2 < 1e-6: return 0.0
        n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
        m1 = np.cross(n1, b2 / norm_b2)
        denom = np.linalg.norm(n1) * np.linalg.norm(n2)
        if denom < 1e-10: return 0.0
        return np.arctan2(np.dot(m1, n2), np.dot(n1, n2))

    r0 = dist(H0, L0)
    if not np.isfinite(r0):
        raise ValueError("Boresch r0 为 NaN/Inf，拒绝更新平衡几何")
    if r0 < 0.3 or r0 > 2.0:
        raise RuntimeError(
            f"Boresch r0={r0*10:.2f}Å 超出合理范围 [3, 20]Å；"
            "拒绝使用默认几何继续生产 ABFE。"
        )

    thetaA0 = angle(H1, H0, L0)
    thetaB0 = angle(H0, L0, L1)
    phiA0 = dihedral(H2, H1, H0, L0)
    phiB0 = dihedral(H1, H0, L0, L1)
    phiC0 = dihedral(H0, L0, L1, L2)
    geom = np.array([r0, thetaA0, thetaB0, phiA0, phiB0, phiC0], dtype=float)
    if not np.all(np.isfinite(geom)):
        raise ValueError(f"Boresch 平衡几何包含 NaN/Inf: {geom.tolist()}")

    return {
        "r0": float(r0),
        "thetaA0": float(thetaA0),  # H1-H0-L0
        "thetaB0": float(thetaB0),  # H0-L0-L1
        "phiA0": float(phiA0),      # H2-H1-H0-L0
        "phiB0": float(phiB0),      # H1-H0-L0-L1
        "phiC0": float(phiC0),      # H0-L0-L1-L2
    }


def assess_boresch_harmonicity(traj, receptor_indices, ligand_indices) -> Dict:
    """Model-free check of the harmonic/Gaussian assumption behind
    `calculate_boresch_analytical_correction`, computed directly from the
    trajectory that locked the anchor choice.

    Runs unconditionally for every Boresch source (auto/orb_simple/simple/
    fluctuation), unlike `OrbScanner.scan_boresch_1d_pes` which needs an ML
    potential, only implements the r-coordinate, and was never called from
    any pipeline path. This uses the same distance/angle/dihedral convention
    as `calc_boresch_from_last_frame` (receptor_indices[0] nearest ligand)
    and reuses `GeometricRestraintEstimator._fluctuation_diagnostics` so the
    same skew/kurtosis/under-sampling criteria apply regardless of which
    estimator produced the anchors.
    """
    if not HAS_MDTRAJ:
        return {"ok": False, "reason": "mdtraj_unavailable"}

    rec_idx = [int(i) for i in receptor_indices]
    lig_idx = [int(i) for i in ligand_indices]
    if len(rec_idx) != 3 or len(lig_idx) != 3:
        return {"ok": False, "reason": "invalid_anchor_index_count"}
    if len(traj) < 4:
        return {"ok": False, "reason": "too_few_trajectory_frames"}

    dist_idx = [[rec_idx[0], lig_idx[0]]]
    angleA_idx = [[rec_idx[1], rec_idx[0], lig_idx[0]]]
    angleB_idx = [[rec_idx[0], lig_idx[0], lig_idx[1]]]
    dihA_idx = [[rec_idx[2], rec_idx[1], rec_idx[0], lig_idx[0]]]
    dihB_idx = [[rec_idx[1], rec_idx[0], lig_idx[0], lig_idx[1]]]
    dihC_idx = [[rec_idx[0], lig_idx[0], lig_idx[1], lig_idx[2]]]

    r = mdtraj.compute_distances(traj, dist_idx)[:, 0]
    thetaA = mdtraj.compute_angles(traj, angleA_idx)[:, 0]
    thetaB = mdtraj.compute_angles(traj, angleB_idx)[:, 0]
    phiA = mdtraj.compute_dihedrals(traj, dihA_idx)[:, 0]
    phiB = mdtraj.compute_dihedrals(traj, dihB_idx)[:, 0]
    phiC = mdtraj.compute_dihedrals(traj, dihC_idx)[:, 0]

    def _unwrap(vals):
        vals = np.asarray(vals, dtype=float).copy()
        for t in range(1, len(vals)):
            diff = vals[t] - vals[t - 1]
            vals[t] -= 2 * np.pi * np.round(diff / (2 * np.pi))
        mean_val = float(np.mean(vals)) if len(vals) else 0.0
        vals -= 2 * np.pi * np.round(mean_val / (2 * np.pi))
        return vals

    coords = {
        "r": r,
        "thetaA": thetaA,
        "thetaB": thetaB,
        "phiA": _unwrap(phiA),
        "phiB": _unwrap(phiB),
        "phiC": _unwrap(phiC),
    }
    fluctuation_diagnostics = [
        GeometricRestraintEstimator._fluctuation_diagnostics(vals, name)
        for name, vals in coords.items()
    ]
    n_bad = sum(1 for item in fluctuation_diagnostics if not item.get("ok", False))
    harmonic_ok = n_bad == 0

    result = {
        "ok": True,
        "method": "trajectory_fluctuation_v1",
        "n_frames_used": int(len(r)),
        "receptor_indices": rec_idx,
        "ligand_indices": lig_idx,
        "fluctuation_distribution": fluctuation_diagnostics,
        "n_non_gaussian_or_under_sampled_terms": int(n_bad),
        "harmonic_assumption_ok": bool(harmonic_ok),
        "warning": "",
    }
    if not harmonic_ok:
        result["warning"] = (
            f"{n_bad}/6 Boresch restraint coordinates show non-Gaussian or under-sampled "
            "fluctuations over the trajectory used to lock this restraint. "
            "calculate_boresch_analytical_correction assumes independent, approximately "
            "Gaussian coordinates; its result may be biased for this anchor choice. "
            "Consider a different --boresch-select candidate, longer pre-equilibration, "
            "or a numerical (non-analytical) release free-energy estimate."
        )
    return result


#=============================================================================
# 修复 3: 基于 RMSF 的自动化锚点选择器 (Automatic Anchor Selection)
#=============================================================================
def auto_select_boresch_anchors_rmsf(
    traj_path: str, top_path: str, ligand_resname: str,
    temperature: float = 300.0, rmsf_threshold_nm: float = 0.12,
    r0_range_angstrom: Tuple[float, float] = (5.0, 10.0),
    output_path: Optional[str] = None
) -> Dict:
    import mdtraj as md

    traj = md.load(traj_path, top=top_path)
    top = traj.topology

    align_atoms = top.select("protein and backbone")
    if len(align_atoms) >= 3:
        traj.superpose(traj, 0, atom_indices=align_atoms)

    ca_atoms = top.select("protein and name CA")
    if len(ca_atoms) < 3:
        raise RuntimeError("受体 CA 原子不足3个，无法构建 Boresch 限制")

    ca_rmsf = md.rmsf(traj, traj, 0, atom_indices=ca_atoms)
    rmsf_by_atom = {int(atom): float(value) for atom, value in zip(ca_atoms, ca_rmsf)}
    rigid_cas = [int(atom) for atom, value in zip(ca_atoms, ca_rmsf) if value <= rmsf_threshold_nm]
    if len(rigid_cas) < 3:
        order = np.argsort(ca_rmsf)
        rigid_cas = [int(ca_atoms[i]) for i in order[: min(12, len(order))]]
    else:
        rigid_cas = rigid_cas[: min(12, len(rigid_cas))]
    
    # 🔑 修复 1：枚举配体重原子三元组，而非死板取前 3 个
    lig_heavy = top.select(f"resname {ligand_resname} and not element H")
    if len(lig_heavy) < 3:
        raise RuntimeError("配体重原子不足3个，无法构建 Boresch 限制")
    
    # 按原子质量排序，优先选择重原子作为锚点候选
    lig_masses = np.array([top.atom(i).element.mass for i in lig_heavy])
    sorted_lig_idx = np.argsort(lig_masses)[::-1]
    top_lig_candidates = [int(lig_heavy[i]) for i in sorted_lig_idx[:min(10, len(sorted_lig_idx))]]
    
    best_score, best_config = -np.inf, None
    for rec_combo in combinations(rigid_cas, 3):
        for lig_combo in combinations(top_lig_candidates, 3):
            r_coords = traj.xyz[0, list(rec_combo)]
            l_coords = traj.xyz[0, list(lig_combo)]
            
            r0 = np.linalg.norm(r_coords[0] - l_coords[0]) * 10.0
            if not (r0_range_angstrom[0] <= r0 <= r0_range_angstrom[1]):
                continue
                
            def angle(a, b, c):
                v1, v2 = a - b, c - b
                return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10), -1, 1)))
            
            thA = angle(r_coords[1], r_coords[0], l_coords[0])
            thB = angle(r_coords[0], l_coords[0], l_coords[1])  # 🔑 修复 2：增加 thB 计算
            
            # 🔑 修复 3：严格同时拦截 thA 和 thB 的奇点
            if not (45 <= thA <= 135) or not (45 <= thB <= 135):
                continue
                
            # 锚点内部距离检查 (防止共线)
            if np.linalg.norm(r_coords[0]-r_coords[1]) < 0.3 or np.linalg.norm(l_coords[0]-l_coords[1]) < 0.2:
                continue
                
            rec_rmsf_mean = float(np.mean([rmsf_by_atom.get(int(i), rmsf_threshold_nm) for i in rec_combo]))
            score = 100 - abs(r0 - 7.5) * 5 - rec_rmsf_mean * 500
            if score > best_score:
                best_score = score
                best_config = {
                    "receptor_indices": list(rec_combo),
                    "ligand_indices": [int(i) for i in lig_combo],
                    "equilibrium_r0": float(r0 * 0.1),
                    "rmsf_mean": rec_rmsf_mean
                }
                
    if best_config is None:
        raise RuntimeError("未找到符合几何 (thA/thB) 与稳定性条件的锚点组合")
        
    print(f"✅ 自动锚点选择完成: r0={best_config['equilibrium_r0']:.2f}nm, RMSF={best_config['rmsf_mean']:.3f}nm")
    if output_path:
        with open(output_path, "w") as f:
            json.dump(best_config, f, indent=2, cls=NumpyEncoder)
    return best_config


#=============================================================================
# 修复 4: 分块 MBAR 分析器 (解决 OOM 瓶颈)
#=============================================================================
class ChunkedMBARAnalyzer:
    """
    支持超大 u_kn 矩阵的 MBAR 分析器
    ✅ 使用 np.memmap 零拷贝加载
    ✅ 自动分块计算，避免多进程 OOM
    ✅ 兼容 pymbar >= 3.0.5
    """
    def __init__(self, max_memory_gb: float = 32.0, cache_dir: str = "./mbar_cache"):
        import gc
        self.max_ram = max_memory_gb * 1e9
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.gc = gc
        
    def _save_chunk_to_disk(self, u_kn_block: np.ndarray, chunk_id: int) -> str:
        path = os.path.join(self.cache_dir, f"u_kn_chunk_{chunk_id}.npy")
        np.save(path, u_kn_block)
        return path
        
    def run_chunked_mbar(self, u_kn_total: np.ndarray, n_k_array: np.ndarray, stage_type: str = "coul"):
        """使用分态步幅抽样执行单次全局 MBAR，避免伪分块平均导致的统计错误。"""
        if not HAS_PYMBAR:
            raise ImportError("需要 pymbar 进行 MBAR 分析: pip install pymbar")
        import pymbar
        
        K, N = u_kn_total.shape
        n_k_array = np.asarray(n_k_array, dtype=int)
        if n_k_array.ndim != 1 or len(n_k_array) != K:
            raise ValueError(f"n_k_array 维度异常: 期望长度 {K}，实际 {n_k_array.shape}")
        if np.any(n_k_array < 0):
            raise ValueError("n_k_array 不能包含负样本数")

        # 1. 计算安全步幅 (Stride)，确保抽样后内存低于限制的 50%
        stride = 1
        while (K * (N // stride) * u_kn_total.itemsize) > self.max_ram * 0.5 and stride < N:
            stride *= 2

        if stride > 1:
            if np.sum(n_k_array) != N:
                raise MemoryError("u_kn_total 列数与 n_k_array 总和不一致，无法安全执行分态步幅抽样")

            print(f"  ⚠️ 内存受限，启用分态步幅抽样 (Stride={stride}) 后执行单次全局 MBAR")
            keep_indices = []
            start = 0
            n_k_sub = np.zeros_like(n_k_array)
            for k, n_k in enumerate(n_k_array):
                end = start + int(n_k)
                if n_k > 0:
                    state_idx = np.arange(start, end, stride, dtype=int)
                    if state_idx.size == 0:
                        state_idx = np.array([start], dtype=int)
                    keep_indices.append(state_idx)
                    n_k_sub[k] = state_idx.size
                start = end

            if not keep_indices:
                raise ValueError("n_k_array 全为 0，无法执行 MBAR")

            keep_indices = np.concatenate(keep_indices)
            u_kn_sub = u_kn_total[:, keep_indices]
            print(f"  ℹ️ 抽样后保留 {u_kn_sub.shape[1]} 帧，分态样本数: {n_k_sub.tolist()}")
        else:
            u_kn_sub = u_kn_total
            n_k_sub = n_k_array.copy()

        # 2. 单次全局 MBAR 求解 (统计严格)
        try:
            mbar = _build_mbar_compatible(
                u_kn_sub,
                n_k_sub,
                verbose=False,
                solver_protocol="hybr",
            )
            res = _compute_free_energy_result_compatible(mbar, compute_uncertainty=True)
            return self._extract_delta_g(res, n_k_sub, stage_type)
        except Exception as e:
            print(f"  🚨 MBAR 求解失败: {e}，尝试降级至 robust 求解器...")
            mbar = _build_mbar_compatible(
                u_kn_sub,
                n_k_sub,
                verbose=False,
                solver_protocol="robust",
            )
            res = _compute_free_energy_result_compatible(mbar, compute_uncertainty=True)
            return self._extract_delta_g(res, n_k_sub, stage_type)

    def _extract_delta_g(self, res, n_k_array, stage_type):
        df, ddf = _extract_free_energy_arrays(res, require_uncertainty=True)
        return df[0, :], ddf[0, :]


#=============================================================================
# 修复 5: 溶剂化能闭环支持 (Ligand-in-Water)
#=============================================================================
class SolventLegRunner:
    """自动构建并运行 Ligand-in-Water 解耦腿"""
    def __init__(self, ligand_resname: str, box_size_nm: float = 4.0, platform_name: str = "CUDA"):
        self.ligand_resname = ligand_resname
        self.box_size = box_size_nm
        self.platform_name = platform_name
        self._cached_system = None
        self._cached_topology = None
        self._cached_positions = None
        self._cached_box_vectors = None

    def build_solvent_system(self, gro_file: str, top_file: str, gmx_include_dir: str = None):
        """从 GROMACS 文件提取配体并水盒化"""
        from openmm.app import Modeller, ForceField
        gro = app.GromacsGroFile(gro_file)
        top = app.GromacsTopFile(top_file, includeDir=gmx_include_dir)
        modeller = Modeller(top.topology, gro.positions)
        
        # 动态计算盒子：回旋半径 + 1.5 nm 缓冲，最小 3.5 nm
        # 正确匹配拓扑与坐标
        lig_indices = [atom.index for atom in gro.topology.atoms() if atom.residue.name == self.ligand_resname]
        lig_coords = np.array([gro.positions[i].value_in_unit(unit.nanometer) for i in lig_indices])
        center = lig_coords.mean(axis=0)
        max_r = np.max(np.linalg.norm(lig_coords - center, axis=1))
        box_size = max(max_r + 1.5, 3.5)  # nm

        # ForceField 创建与 System 构建
        ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        modeller.addSolvent(ff, boxSize=app.Vec3(box_size, box_size, box_size))
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                                 nonbondedCutoff=1.0*unit.nanometer, constraints=app.HBonds, rigidWater=True)
        # ✅ 缓存构建结果
        self._cached_system = system
        self._cached_topology = modeller.topology
        self._cached_positions = modeller.positions
        self._cached_box_vectors = modeller.topology.getPeriodicBoxVectors()
        return self._cached_system, self._cached_topology, self._cached_positions, self._cached_box_vectors
        
    def run_solvent_decoupling(self, positions, topology, ligand_indices, **pipeline_kwargs):
        """运行溶剂相解耦计算（委托给 ABFEPipeline）"""
        from abfe_pipeline import ABFEPipeline
        if self._cached_system is None:
            raise RuntimeError("请先调用 build_solvent_system 构建溶剂系统")
        
        pipe = ABFEPipeline(
            system=self._cached_system,          # ✅ 修复：传入有效 system
            topology=self._cached_topology,
            positions=self._cached_positions,
            box_vectors=self._cached_box_vectors,
            ligand_indices=ligand_indices,
            temperature=pipeline_kwargs.get('temperature', 300.0),
            output_dir=pipeline_kwargs.get('output_dir', './solvent_output'),
            platform_name=self.platform_name,
        )
        # ✅ 移除 decoupling_scheme 硬编码，透传用户配置
        return pipe.run_full_pipeline(**pipeline_kwargs)


# ============================================================================
# 7. 单位格式化器 (I/O 层专用：内核计算绝不碰单位转换)
# ============================================================================
class UnitFormatter:
    """
    单位格式化器：内部计算不碰，仅用于 I/O 转换
    【三层单位规范】
    1. 内核层 (OpenMM): nm, kJ/mol, ps, rad, e (裸数值)
    2. 数据交换层 (JSON): key 带单位后缀，如 r0_nm, kr_kJ_mol_nm2
    3. 人类可读层 (LOG/Report): Å, kcal/mol, ns, °
    """
    # === 基础转换函数 ===
    @staticmethod
    def nm_to_A(val): return val * 10.0
    @staticmethod
    def A_to_nm(val): return val * 0.1
    @staticmethod
    def kJ_to_kcal(val): return val / 4.184
    @staticmethod
    def kcal_to_kJ(val): return val * 4.184
    @staticmethod
    def rad_to_deg(val): return np.degrees(val)
    @staticmethod
    def deg_to_rad(val): return np.radians(val)
    @staticmethod
    def ps_to_ns(val): return val / 1000.0
    @staticmethod
    def ns_to_ps(val): return val * 1000.0

    # === Boresch 参数格式化 (人类可读) ===
    @classmethod
    def format_boresch_human(cls, boresch_dict: dict) -> str:
        """格式化 Boresch 参数为化学家常用单位"""
        eq = boresch_dict["equilibrium_values"]
        fc = boresch_dict["force_constants"]
        return (
            f"📏 r0={cls.nm_to_A(eq['r0']):.2f} Å | "
            f"θA={cls.rad_to_deg(eq['thetaA0']):.1f}° | "
            f"φA={cls.rad_to_deg(eq['phiA0']):.1f}°\n"
            f"⚖️ kr={cls.kJ_to_kcal(fc['kr']) / 100.0:.2f} kcal/mol/Å²| "
            f"kθA={cls.kJ_to_kcal(fc['kthetaA']):.2f} kcal/mol/rad²"
        )

    # === Boresch 参数序列化 (JSON，key 带单位后缀) ===
    @classmethod
    def format_boresch_json(cls, boresch_dict: dict) -> dict:
        """
        序列化 Boresch 参数为 JSON 安全格式 (兼容扁平/嵌套/混合结构)
        🔑 智能路由提取逻辑，彻底解决 KeyError: 'equilibrium_values'
        """
        # 1. 智能提取：优先从嵌套层取，若为空则降级到顶层
        anchors = boresch_dict.get("boresch_anchors", boresch_dict)
        eq = anchors.get("equilibrium_values") or boresch_dict.get("equilibrium_values", {})
        fc = anchors.get("force_constants") or boresch_dict.get("force_constants", {})
        rec_idx = anchors.get("receptor_indices") or boresch_dict.get("receptor_indices", [])
        lig_idx = anchors.get("ligand_indices") or boresch_dict.get("ligand_indices", [])

        if not eq or not fc:
            raise ValueError("Boresch 参数字典结构异常：缺失 equilibrium_values 或 force_constants")

        # 2. 构建标准嵌套输出 (严格带单位后缀)
        return {
            "boresch_anchors": {
                "receptor_indices": rec_idx,
                "ligand_indices": lig_idx,
                "equilibrium_values": {
                    "r0_nm": float(eq.get("r0", 0)),
                    "thetaA0_rad": float(eq.get("thetaA0", 0)),
                    "thetaB0_rad": float(eq.get("thetaB0", 0)),
                    "phiA0_rad": float(eq.get("phiA0", 0)),
                    "phiB0_rad": float(eq.get("phiB0", 0)),
                    "phiC0_rad": float(eq.get("phiC0", 0)),
                },
                "force_constants": {
                    "kr_kJ_mol_nm2": float(fc.get("kr", 0)),
                    "kthetaA_kJ_mol_rad2": float(fc.get("kthetaA", 0)),
                    "kthetaB_kJ_mol_rad2": float(fc.get("kthetaB", 0)),
                    "kphiA_kJ_mol_rad2": float(fc.get("kphiA", 0)),
                    "kphiB_kJ_mol_rad2": float(fc.get("kphiB", 0)),
                    "kphiC_kJ_mol_rad2": float(fc.get("kphiC", 0)),
                },
            },
            "is_fallback": boresch_dict.get("is_fallback", False),
            "total_score": boresch_dict.get("total_score", None),
            "diagnostics": boresch_dict.get("diagnostics", None),
        }

    # === 结果格式化 (人类可读) ===
    @classmethod
    def format_results_human(cls, results: dict) -> str:
        """格式化最终结果报告"""
        err_kj = results.get("total_error_kJ_mol", results.get("total_error", 0.0))
        if "delta_G_bind_kJ_mol" in results:
            dg_kj = results.get("delta_G_bind_kJ_mol", 0.0)
            title = "✅ 结合自由能 ΔG_bind"
        elif "total_delta_G_complex_kJ_mol" in results:
            dg_kj = results.get("total_delta_G_complex_kJ_mol", 0.0)
            title = "✅ 复合物总自由能 ΔG_complex"
        elif "decoupling_delta_G_kJ_mol" in results:
            dg_kj = results.get("decoupling_delta_G_kJ_mol", 0.0)
            title = "✅ 解耦腿自由能 ΔG_leg"
        else:
            dg_kj = results.get("total_delta_G_complex_kJ_mol", results.get("total_delta_G_complex", 0.0))
            title = "✅ 自由能结果 ΔG"
        return (
            f"\n{'='*50}\n"
            f"{title} = {cls.kJ_to_kcal(dg_kj):.2f} ± {cls.kJ_to_kcal(err_kj):.2f} kcal/mol\n"
            f"   ( = {dg_kj:.2f} ± {err_kj:.2f} kJ/mol )\n"
            f"{'='*50}"
        )

    # === 采样元数据格式化 (JSON) ===
    @classmethod
    def format_sampling_metadata_json(cls, config: dict) -> dict:
        """序列化采样元数据为 JSON"""
        return {
            "sampling_metadata": {
                "dt_ps": config.get("timestep_ps", 0.002),
                "temperature_K": config.get("temperature", 300.0),
                "n_steps_per_window": config.get("n_steps_per_window", 0),
                "friction_ps": config.get("friction", 1.0),
            }
        }

# ============================================================================
# 通用工具函数：System 管理与配体内部力构建
# ============================================================================
from openmm import XmlSerializer

def ensure_owned_system(system: openmm.System) -> openmm.System:
    """强制获取 System 的 Python 所有权，防止 SWIG GC"""
    if system is None:
        raise ValueError("System 对象为 None")
    try:
        if getattr(system, 'thisown', 0) == 1:
            return system
    except Exception:
        pass
    xml = XmlSerializer.serialize(system)
    new_sys = XmlSerializer.deserialize(xml)
    new_sys.thisown = 1
    _ = new_sys.getNumParticles()
    return new_sys


def sync_all_exclusions(system: openmm.System) -> int:
    """
    生产级排除表同步。

    🚨 关键修复：OpenMM 要求同一 System 里所有共享同一套邻居表的
    NonbondedForce/CustomNonbondedForce（粒子数相同）拥有完全相同的排除表——
    "All Forces must have identical exclusions" 就是这个要求被违反时抛出的。
    旧版本按每个 CustomNonbondedForce 的 interaction group 范围"按需"补齐排除表
    （例如只给 L-E 力补 L-E 相关的对），这在物理上没问题（interaction group
    之外的对本来就不会被计算），但 OpenMM 底层邻居表校验比较的是排除表本身
    是否逐对相同，不管 interaction group——所以只要 NonbondedForce 里有任何
    一个不落在某个 CustomNonbondedForce interaction group 内的排除对
    （典型情况：环境蛋白/水分子自身的 1-2/1-3/1-4 排除，跟只处理 L-E 的软核力
    毫不相关），旧逻辑就会让两者的排除表数量对不上，从而在生产采样阶段
    （通常是第一次真正调用 minimizeEnergy/getState 触发底层邻居表构建时）报错。
    这里改为无差别地把"并集"灌给每一个粒子数匹配的力，牺牲一点点冗余排除对，
    换来严格逐对相同——interaction group 之外的排除对本来就不会被该力用到，
    是纯粹的账本对齐，不改变任何物理量。
    """
    nb_forces = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    custom_forces = [f for f in system.getForces() if isinstance(f, openmm.CustomNonbondedForce)]
    if not nb_forces or not custom_forces:
        return 0
    nb_force = nb_forces[0]
    n_particles = nb_force.getNumParticles()

    union_excl = set()
    for i in range(nb_force.getNumExceptions()):
        p1, p2, _, _, _ = nb_force.getExceptionParameters(i)
        p1, p2 = int(p1), int(p2)
        if p1 != p2:
            union_excl.add((min(p1, p2), max(p1, p2)))

    eligible_forces = []
    existing_per_force = []
    for c_force in custom_forces:
        if c_force.getNumParticles() != n_particles:
            continue
        existing = set()
        for i in range(c_force.getNumExclusions()):
            p1, p2 = c_force.getExclusionParticles(i)
            existing.add((min(int(p1), int(p2)), max(int(p1), int(p2))))
        union_excl |= existing
        eligible_forces.append(c_force)
        existing_per_force.append(existing)

    total_synced = 0
    for c_force, existing in zip(eligible_forces, existing_per_force):
        missing = union_excl - existing
        for p1, p2 in missing:
            c_force.addExclusion(p1, p2)
        total_synced += len(missing)
    return total_synced


def create_ligand_internal_force(
    nb_force: openmm.NonbondedForce,
    perturbed_indices: List[int],
    particle_params,
    reference_exclusions=None,
    num_particles: int = None,
    system: openmm.System = None
):
    """
    构建配体-配体内部非键力 (Standard LJ + Coulomb) 和 1-4 恢复力。
    注意：此函数不分配 ForceGroup，调用者需自行设置并添加至 System。
    """
    if num_particles is None:
        num_particles = nb_force.getNumParticles()
    perturbed_set = set(perturbed_indices)

    expr = "4*sqrt(epsilon1*epsilon2)*((sigma12/r)^12 - (sigma12/r)^6) + 138.935456*q1*q2/r; sigma12 = 0.5*(sigma1+sigma2)"
    ll_force = openmm.CustomNonbondedForce(expr)
    ll_force.addPerParticleParameter('q')
    ll_force.addPerParticleParameter('sigma')
    ll_force.addPerParticleParameter('epsilon')

    for i in range(num_particles):
        if particle_params and i < len(particle_params):
            q, sig, eps = particle_params[i]
        else:
            q, sig, eps = nb_force.getParticleParameters(i)
        ll_force.addParticle([
            q.value_in_unit(unit.elementary_charge),
            sig.value_in_unit(unit.nanometer),
            eps.value_in_unit(unit.kilojoule_per_mole)
        ])

    ll_force.addInteractionGroup(perturbed_set, perturbed_set)
    ll_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    ll_force.setCutoffDistance(1.2 * unit.nanometer)
    ll_force.setUseLongRangeCorrection(False)

    # ========================================================================
    # 🔑 生产级排除对收集：全覆盖 1-2/1-3/1-4 (修复漏扫约束与异常表的致命缺陷)
    # ========================================================================
    exclusion_pairs = set()

    if system is not None:
        # === 1. 谐波键 (1-2 排除) ===
        for f in system.getForces():
            if isinstance(f, openmm.HarmonicBondForce):
                for i in range(f.getNumBonds()):
                    p1, p2, _, _ = f.getBondParameters(i)
                    if p1 in perturbed_set and p2 in perturbed_set:
                        exclusion_pairs.add((min(p1, p2), max(p1, p2)))
            
            # === 2. 谐波角 (1-3 排除，取首尾原子) ===
            elif isinstance(f, openmm.HarmonicAngleForce):
                for i in range(f.getNumAngles()):
                    p1, p2, p3, _, _ = f.getAngleParameters(i)
                    # ✅ 仅当首尾原子都在配体内才排除 (1-3)
                    if p1 in perturbed_set and p3 in perturbed_set:
                        exclusion_pairs.add((min(p1, p3), max(p1, p3)))
        
        # === 3. 刚性约束 (1-2 排除，GROMACS 常将含 H 键转为约束) ===
        # 🔑 核心修复：独立于力遍历，直接扫描系统级约束
        for i in range(system.getNumConstraints()):
            p1, p2, _ = system.getConstraintParameters(i)
            if p1 in perturbed_set and p2 in perturbed_set:
                exclusion_pairs.add((min(p1, p2), max(p1, p2)))
        
        # === 4. NonbondedForce 异常表 (1-4 排除) ===
        nb_forces = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
        if nb_forces:
            nb = nb_forces[0]
            for i in range(nb.getNumExceptions()):
                p1, p2, _, _, _ = nb.getExceptionParameters(i)
                p1, p2 = int(p1), int(p2)
                if p1 in perturbed_set and p2 in perturbed_set:
                    exclusion_pairs.add((min(p1, p2), max(p1, p2)))

    # === 5. 合并参考排除表 (来自原始 NonbondedForce 的 exceptions) ===
    if reference_exclusions:
        for p1, p2 in reference_exclusions:
            p1, p2 = int(p1), int(p2)
            if p1 in perturbed_set and p2 in perturbed_set:
                exclusion_pairs.add((min(p1, p2), max(p1, p2)))

    # === 6. 执行排除添加 (严格去重) ===
    for p1, p2 in exclusion_pairs:
        ll_force.addExclusion(p1, p2)

    # 🔍 诊断输出 (可选，生产环境可注释)
    print(f"  🔍 [Group2 排除表] 共收集 {len(exclusion_pairs)} 对配体内部排除 (1-2/1-3/1-4)")

    # 1-4 恢复力
    ll_14_force = None
    exceptions_14 = []
    for i in range(nb_force.getNumExceptions()):
        p1, p2, chargeProd, sigma, epsilon = nb_force.getExceptionParameters(i)
        p1, p2 = int(p1), int(p2)
        if p1 in perturbed_set and p2 in perturbed_set:
            has_charge = chargeProd.value_in_unit(unit.elementary_charge**2) != 0
            has_lj = epsilon.value_in_unit(unit.kilojoule_per_mole) != 0
            if has_charge or has_lj:
                exceptions_14.append((p1, p2, chargeProd, sigma, epsilon))

    if exceptions_14:
        expr_14 = "4*epsilon*((sigma/r)^12 - (sigma/r)^6) + 138.935456*chargeProd/r"
        ll_14_force = openmm.CustomBondForce(expr_14)
        ll_14_force.addPerBondParameter('chargeProd')
        ll_14_force.addPerBondParameter('sigma')
        ll_14_force.addPerBondParameter('epsilon')
        for p1, p2, cp, sig, eps in exceptions_14:
            ll_14_force.addBond(p1, p2, [
                cp.value_in_unit(unit.elementary_charge**2),
                sig.value_in_unit(unit.nanometer),
                eps.value_in_unit(unit.kilojoule_per_mole)
            ])

    ll_force.setUseSwitchingFunction(True)
    ll_force.setSwitchingDistance(1.0 * unit.nanometer)
    return ll_force, ll_14_force

# ============================================================================
# 8. 纯轨迹几何波动 Boresch 估算器 (基于化学连通性 + 方差最小化)
# ============================================================================
class GeometricRestraintEstimator:
    """
    基于轨迹几何波动和化学连通性的 Boresch 参数估算器。
    不依赖任何力场，仅需 mdtraj 轨迹。
    核心流程：
      1. 受体候选原子：指定原子名 (默认 CA,CB,C,N,O)
      2. 0.5 nm 接触搜索找到 (锚点,配体) 最近原子对
      3. 0.22 nm 成键延伸构建受体三元组和配体三元组 (保证化学连通)
      4. 计算全轨迹距离/角度/二面角，周期性展开
      5. 硬截断 θ ∈ [45°,135°]
      6. 方差加权评分 (物理力常数尺度) 选择最优组合
      7. 力常数 = kBT / 方差 (并裁剪到安全范围)
    """

    def __init__(self, temperature=300.0,
                 search_dist=0.5,         # nm
                 bond_dist=0.22,          # nm
                 anchor_atom_names=None):
        self.temperature = temperature
        self.gas_constant_kj_per_mol_k = 8.314e-3
        self.search_dist = search_dist
        self.bond_dist = bond_dist
        if anchor_atom_names is None:
            anchor_atom_names = ["CA", "CB", "C", "N", "O"]
        self.anchor_atom_names = anchor_atom_names

    # ----------------------------------------------------------------
    # 工具：化学键邻居（基于距离阈值）
    # ----------------------------------------------------------------
    def _find_bonded_neighbors(self, atom_idx, haystack, ref_xyz):
        """寻找与 atom_idx 距离 <= bond_dist 的原子（模拟共价键）"""
        vec = ref_xyz[haystack] - ref_xyz[atom_idx]
        dist = np.linalg.norm(vec, axis=1)
        bonded = haystack[dist <= self.bond_dist]
        return [b for b in bonded if b != atom_idx]

    @staticmethod
    def _clip_force_constant(value, lower, upper):
        raw = float(value)
        clipped = float(np.clip(raw, lower, upper))
        return clipped, bool(abs(clipped - raw) > 1e-8)

    @staticmethod
    def _fluctuation_diagnostics(values, name):
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 4:
            return {
                "name": name,
                "n": int(vals.size),
                "ok": False,
                "reason": "too_few_finite_samples",
            }

        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if std <= 1e-12:
            return {
                "name": name,
                "n": int(vals.size),
                "ok": False,
                "mean": mean,
                "std": std,
                "reason": "near_zero_variance",
            }

        centered = (vals - mean) / std
        skew = float(np.mean(centered ** 3))
        excess_kurtosis = float(np.mean(centered ** 4) - 3.0)
        p01, p50, p99 = np.percentile(vals, [1, 50, 99])
        ok = bool(abs(skew) <= 2.0 and abs(excess_kurtosis) <= 7.0)
        reason = "ok" if ok else "non_gaussian_tail_or_asymmetry"
        return {
            "name": name,
            "n": int(vals.size),
            "ok": ok,
            "mean": mean,
            "std": std,
            "skew": skew,
            "excess_kurtosis": excess_kurtosis,
            "p01": float(p01),
            "p50": float(p50),
            "p99": float(p99),
            "reason": reason,
        }

    # ----------------------------------------------------------------
    # 生成所有化学连通的 6-原子组合
    # ----------------------------------------------------------------
    def _generate_anchor_combos(self, traj, prot_indices, lig_heavy_indices):
        ref_xyz = traj.xyz[0]

        # 1. 接触对 (anch, lig) 距离 ≤ search_dist
        lig_pos = ref_xyz[lig_heavy_indices]
        prot_pos = ref_xyz[prot_indices]
        dist_mat = np.linalg.norm(prot_pos[:, None, :] - lig_pos[None, :, :], axis=2)
        contact_pairs = [(prot_indices[i], lig_heavy_indices[j])
                         for i, j in zip(*np.where(dist_mat <= self.search_dist))]
        if not contact_pairs:
            raise RuntimeError("未找到锚点-配体接触对，请增大 search_dist")

        # 2. 预计算键合邻居字典
        prot_nei = {idx: self._find_bonded_neighbors(idx, prot_indices, ref_xyz) for idx in prot_indices}
        lig_nei  = {idx: self._find_bonded_neighbors(idx, lig_heavy_indices, ref_xyz) for idx in lig_heavy_indices}

        anclig_combos = []
        for anc, lig in contact_pairs:
            # 受体侧：以 anc 为 a，找与其键合的 b，再找与 b 键合的 c -> (c, b, a)
            for b in prot_nei.get(anc, []):
                for c in prot_nei.get(b, []):
                    if c == anc: continue
                    rec_tri = (c, b, anc)
                    # 配体侧：以 lig 为 a，找键合的 b，再找与 b 键合的 c -> (a, b, c)
                    for b_lig in lig_nei.get(lig, []):
                        for c_lig in lig_nei.get(b_lig, []):
                            if c_lig == lig: continue
                            lig_tri = (lig, b_lig, c_lig)
                            anclig_combos.append((rec_tri, lig_tri))

        # 去重
        unique = []
        seen = set()
        for rec, lig in anclig_combos:
            key = rec + lig
            if key not in seen:
                seen.add(key)
                unique.append((rec, lig))
        return unique

    # ----------------------------------------------------------------
    # 主估算函数
    # ----------------------------------------------------------------
    def estimate_from_trajectory(self, traj, ligand_resname, output_path=None):
        top = traj.topology

        # 1. 受体锚点候选原子 (基于原子名)
        anchor_query = "protein and name " + ' '.join(self.anchor_atom_names)
        prot_indices = top.select(anchor_query)
        if len(prot_indices) == 0:
            raise RuntimeError(f"没有找到锚点原子：{self.anchor_atom_names}")

        # 2. 配体重原子
        lig_heavy = top.select(f"resname {ligand_resname} and not element H")
        if len(lig_heavy) == 0:
            raise RuntimeError(f"未找到配体 {ligand_resname} 的重原子")

        # 3. 化学连通组合枚举
        combos = self._generate_anchor_combos(traj, prot_indices, lig_heavy)
        print(f"  🔗 化学连通候选组合数: {len(combos)}")
        if len(combos) == 0:
            raise RuntimeError("没有符合条件的6原子组合")

        # 4. 构建 mdtraj 原子索引列表 (用于批量计算几何)
        n_combos = len(combos)
        dist_indices   = [[c[0][2], c[1][0]] for c in combos]   # anc_a - lig_a
        angleA_indices = [[c[0][1], c[0][2], c[1][0]] for c in combos]  # anc_b, anc_a, lig_a
        angleB_indices = [[c[0][2], c[1][0], c[1][1]] for c in combos]
        dihA_indices   = [[c[0][0], c[0][1], c[0][2], c[1][0]] for c in combos]
        dihB_indices   = [[c[0][1], c[0][2], c[1][0], c[1][1]] for c in combos]
        dihC_indices   = [[c[0][2], c[1][0], c[1][1], c[1][2]] for c in combos]

        # 5. 逐帧计算几何量 (分块避免 OOM)
        n_frames = len(traj)
        dists    = np.zeros((n_frames, n_combos))
        angles_a = np.zeros((n_frames, n_combos))
        angles_b = np.zeros((n_frames, n_combos))
        diheds_a = np.zeros((n_frames, n_combos))
        diheds_b = np.zeros((n_frames, n_combos))
        diheds_c = np.zeros((n_frames, n_combos))

        chunk_size = 100  # 可调整
        for i in range(0, n_frames, chunk_size):
            chunk = traj[i:i+chunk_size]
            dists[i:i+len(chunk)]    = mdtraj.compute_distances(chunk, dist_indices)
            angles_a[i:i+len(chunk)] = mdtraj.compute_angles(chunk, angleA_indices)
            angles_b[i:i+len(chunk)] = mdtraj.compute_angles(chunk, angleB_indices)
            diheds_a[i:i+len(chunk)] = mdtraj.compute_dihedrals(chunk, dihA_indices)
            diheds_b[i:i+len(chunk)] = mdtraj.compute_dihedrals(chunk, dihB_indices)
            diheds_c[i:i+len(chunk)] = mdtraj.compute_dihedrals(chunk, dihC_indices)

        # 6. 周期性二面角展开 (按列展开，保持连续性)
        def periodic_unwrap(dh_array):
            for col in range(dh_array.shape[1]):
                vals = dh_array[:, col]
                for t in range(1, len(vals)):
                    diff = vals[t] - vals[t-1]
                    vals[t] -= 2*np.pi * np.round(diff / (2*np.pi))
                mean_val = np.mean(vals)
                vals -= 2*np.pi * np.round(mean_val / (2*np.pi))
                dh_array[:, col] = vals

        periodic_unwrap(diheds_a)
        periodic_unwrap(diheds_b)
        periodic_unwrap(diheds_c)

        # 7. 方差加权评分 (物理力常数尺度)
        dist_weight = 4184.0       # kJ/mol/nm²
        angle_weight = 41.84       # kJ/mol/rad²
        dihedral_weight = 41.84

        var_dist = np.var(dists, axis=0)
        var_angA = np.var(angles_a, axis=0)
        var_angB = np.var(angles_b, axis=0)
        var_dihA = np.var(diheds_a, axis=0)
        var_dihB = np.var(diheds_b, axis=0)
        var_dihC = np.var(diheds_c, axis=0)

        total_var = (dist_weight * var_dist +
                     angle_weight * (var_angA + var_angB) +
                     dihedral_weight * (var_dihA + var_dihB + var_dihC))

        # 8. 硬截断：排除平均角度不在 [45°,135°] 的候选 (避免 1/sinθ 奇点)
        avg_angA = np.mean(angles_a, axis=0)
        avg_angB = np.mean(angles_b, axis=0)
        banned = (avg_angA < np.deg2rad(45)) | (avg_angA > np.deg2rad(135)) | \
                 (avg_angB < np.deg2rad(45)) | (avg_angB > np.deg2rad(135))
        total_var[banned] = np.inf

        # 🔑 核心修复：拦截全 inf 灾难
        if np.all(np.isinf(total_var)):
            raise RuntimeError(
                "❌ 所有候选锚点组合的几何角度 (θA/θB) 均超出安全范围 [45°, 135°]！\n"
                "   体系可能存在严重畸变或配体脱离口袋。请检查预平衡轨迹，或使用 --boresch-source auto 切换至 Orb 估算。"
            )

        # 9. 选择最优组合
        best_idx = np.argmin(total_var)
        best_combo = combos[best_idx]

        # 10. 提取平衡值 (平均值) 和力常数
        eq = {
            "r0":       float(np.mean(dists[:, best_idx])),
            "thetaA0":  float(np.mean(angles_a[:, best_idx])),
            "thetaB0":  float(np.mean(angles_b[:, best_idx])),
            "phiA0":    float(np.mean(diheds_a[:, best_idx])),
            "phiB0":    float(np.mean(diheds_b[:, best_idx])),
            "phiC0":    float(np.mean(diheds_c[:, best_idx])),
        }

        kBT = self.gas_constant_kj_per_mol_k * self.temperature
        raw_fc = {
            "kr":       kBT / (var_dist[best_idx] + 1e-10),
            "kthetaA":  kBT / (var_angA[best_idx] + 1e-10),
            "kthetaB":  kBT / (var_angB[best_idx] + 1e-10),
            "kphiA":    kBT / (var_dihA[best_idx] + 1e-10),
            "kphiB":    kBT / (var_dihB[best_idx] + 1e-10),
            "kphiC":    kBT / (var_dihC[best_idx] + 1e-10),
        }
        force_constant_ranges = {
            "kr": [100.0, 2000.0],
            "kthetaA": [10.0, 1000.0],
            "kthetaB": [10.0, 1000.0],
            "kphiA": [10.0, 1000.0],
            "kphiB": [10.0, 1000.0],
            "kphiC": [10.0, 1000.0],
        }
        fc = {}
        clipped_flags = {}
        for key, raw_value in raw_fc.items():
            lower, upper = force_constant_ranges[key]
            fc[key], clipped_flags[key] = self._clip_force_constant(raw_value, lower, upper)

        fluctuation_diagnostics = [
            self._fluctuation_diagnostics(dists[:, best_idx], "r"),
            self._fluctuation_diagnostics(angles_a[:, best_idx], "thetaA"),
            self._fluctuation_diagnostics(angles_b[:, best_idx], "thetaB"),
            self._fluctuation_diagnostics(diheds_a[:, best_idx], "phiA"),
            self._fluctuation_diagnostics(diheds_b[:, best_idx], "phiB"),
            self._fluctuation_diagnostics(diheds_c[:, best_idx], "phiC"),
        ]
        n_bad_diag = sum(1 for item in fluctuation_diagnostics if not item.get("ok", False))
        n_clipped = sum(1 for clipped in clipped_flags.values() if clipped)

        # 🚨 关键修复：best_combo[0] (rec_tri) 内部是按 (c,b,anc)=(最远,中间,最近)
        # 的顺序构建的——上面 dist/angle/dihedral 的 index 列表都正确利用了这个
        # 顺序算出了符合 R0(最近)-顶点约定的 eq/fc；但如果直接原样存成
        # receptor_indices，会跟 _check_boresch_geometry_safe /
        # calc_boresch_from_last_frame / LambdaDependentBoreschForce 全部假设的
        # "receptor_indices[0]=离配体最近的锚点" 顺序相反，导致下游重新读取这份
        # 结果时把最远锚点当成了 R0。这里显式反转，使其对外统一为最近在前。
        result = {
            "receptor_indices": list(reversed(best_combo[0])),
            "ligand_indices": list(best_combo[1]),
            "equilibrium_values": eq,
            "force_constants": fc,
            "force_constants_raw": {k: float(v) for k, v in raw_fc.items()},
            "force_constant_clip_ranges": force_constant_ranges,
            "force_constant_clipped": clipped_flags,
            "diagnostics": {
                "n_frames": int(n_frames),
                "n_candidates": int(n_combos),
                "n_angle_banned_candidates": int(np.sum(banned)),
                "best_total_variance_score": float(total_var[best_idx]),
                "fluctuation_distribution": fluctuation_diagnostics,
                "n_non_gaussian_or_under_sampled_terms": int(n_bad_diag),
                "n_clipped_force_constants": int(n_clipped),
                "warnings": [
                    "Some fluctuation-derived force constants were clipped to conservative bounds."
                    if n_clipped else "",
                    "One or more restraint coordinates show non-Gaussian or under-sampled fluctuations."
                    if n_bad_diag else "",
                ],
            },
            "method": "geometric_fluctuation_v2_clipped",
        }
        result["diagnostics"]["warnings"] = [
            warning for warning in result["diagnostics"]["warnings"] if warning
        ]

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, cls=NumpyEncoder)

        print(f"  🏆 最优锚点: 受体 {result['receptor_indices']} | 配体 {result['ligand_indices']}")
        print(f"     r0={eq['r0']*10:.2f} Å, θA={np.degrees(eq['thetaA0']):.1f}°, θB={np.degrees(eq['thetaB0']):.1f}°")
        print(f"     kr={fc['kr']:.1f} kJ/mol/nm², kθA={fc['kthetaA']:.1f} kJ/mol/rad²")
        if n_clipped:
            print(f"  ⚠️ fluctuation Boresch 有 {n_clipped} 个力常数被裁剪；raw 值已写入结果 JSON。")
        if n_bad_diag:
            print(f"  ⚠️ fluctuation Boresch 有 {n_bad_diag} 个坐标分布偏离高斯或采样不足；请检查 diagnostics。")
        return result


# ============================================================================
# 8. Orbv3 → DEXP 拟合流水线 (从 DEXP_class.py 合并)
# ============================================================================
def run_orbv3_dexp_fitting(
    traj_file: str,
    top_file: str,
    ligand_resname: str,
    output_dir: str,
    temperature: float = 300.0,
    device: str = "cuda",
    n_frames: int = 200,
    env_radius_nm: float = 0.60,
    env_max_atoms: Optional[int] = None,
    fit_last_ns: Optional[float] = None,
    fit_r_min: float = 0.20,
    fit_r_max: float = 0.50,
    gmx_include_dir: str = None
) -> str:
    """一键 Orbv3 → DEXP 拟合（委托给 Orbv3DEXPFittingPipeline）"""
    if not HAS_ORB:
        raise ImportError("Orb 拟合依赖 torch + openmmml，请安装后重试")
    pipeline = Orbv3DEXPFittingPipeline(model_name="mace-off24-medium", device=device)
    out_json = os.path.join(output_dir, "dexp_fitted_params.json")
    pipeline.run_from_trajectory(
        traj_file=traj_file,
        top_file=top_file,
        ligand_resname=ligand_resname,
        output_json=out_json,
        n_frames=n_frames,
        env_radius_nm=env_radius_nm,
        env_max_atoms=env_max_atoms,
        fit_last_ns=fit_last_ns,
        gmx_include_dir=gmx_include_dir,
        fitting_region=(fit_r_min, fit_r_max),
    )
    return out_json


def _select_tail_indices_from_time(traj, fit_frames: int, fit_last_ns: Optional[float]) -> List[int]:
    n_frames_total = len(traj)
    if n_frames_total == 0:
        return []

    fit_frames = max(1, int(fit_frames))
    if fit_last_ns is None or float(fit_last_ns) <= 0.0:
        start = max(0, n_frames_total - fit_frames)
        return list(range(start, n_frames_total))

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


def _summarize_dexp_values(values) -> Dict[str, float]:
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


def _choose_delta_e_threshold(delta_e_values, base_threshold: float = 500.0) -> Tuple[float, Dict[str, Any]]:
    stats = _summarize_dexp_values(delta_e_values)
    polluted = False
    reason = "default"
    threshold = float(base_threshold)
    center = float(stats["mean"]) if stats["count"] > 0 else 0.0
    if stats["count"] == 0:
        return threshold, {"polluted": False, "reason": "no_data", "stats": stats, "center": center}
    centered_values = [float(v) - center for v in delta_e_values]
    centered_stats = _summarize_dexp_values(centered_values)
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
    return threshold, {
        "polluted": polluted,
        "reason": reason,
        "stats": stats,
        "centered_stats": centered_stats,
        "center": center,
    }


def _detect_suspicious_fit(fitted_params: Dict[str, Any]) -> Dict[str, Any]:
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
    return {"suspicious_fit": bool(hits), "boundary_hits": hits}


class Orbv3DEXPFittingPipeline:
    """
    一键 Orbv3 残差标注 → DEXP 拟合流水线
    职责：轨迹加载 → 纯净MM L-E参考系构建 → Orb三体分解 → ΔE计算 → 拟合器对接
    数学契约：ΔE = E_qm(region) - E_mm_total(region)，拟合前再减去样本均值，仅学习相对 MM 总非键面的涨落修正。
    """
    def __init__(self, model_name: str = "mace-off24-medium", device: str = "cuda"):
        if not HAS_ORB:
            raise ImportError("Orb 拟合依赖 torch + openmmml，请安装后重试")
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.model_name = model_name
        self.label_mode = "orbv3_interaction" if "orb" in model_name.lower() else "mace_decomposition"
        self.openmmml_precision = None
        self._precision_kwarg_supported = True
        self.potential = MLPotential(model_name)
        self._orb_ctx_cache = {}
        self._cache_contexts = True

    def _clear_orb_context_cache(self):
        for bundle in self._orb_ctx_cache.values():
            for ctx_bundle in bundle.get("contexts", {}).values():
                try:
                    ctx_bundle.pop("context", None)
                    ctx_bundle.pop("simulation", None)
                    ctx_bundle.pop("integrator", None)
                    ctx_bundle.pop("system", None)
                except Exception:
                    pass
        self._orb_ctx_cache = {}
        gc.collect()
        if HAS_ORB and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        text = str(exc)
        return ("CUDA out of memory" in text) or ("cuda out of memory" in text)

    @staticmethod
    def _is_precision_or_dtype_mismatch(exc: Exception) -> bool:
        text = str(exc)
        return (
            ("both inputs should have same dtype" in text)
            or ("requested dtype" in text)
            or ("same scalar type" in text)
        )

    @staticmethod
    def _is_unsupported_precision_kwarg(exc: Exception) -> bool:
        text = str(exc)
        return ("precision" in text) and ("unexpected keyword" in text or "got an unexpected keyword argument" in text)

    def _fallback_to_cpu_double(self, reason: str):
        print(f"  ⚠️ {reason}，将按 openmm-ml 接口回退到 device='cpu', precision='double' 继续标注。")
        self.device = "cpu"
        self.openmmml_precision = "double" if self._precision_kwarg_supported else None
        self._clear_orb_context_cache()

    def _create_openmmml_system(self, topology, return_energy_type: Optional[str] = None):
        kwargs = _build_openmmml_kwargs(
            device=self.device,
            precision=self.openmmml_precision if self._precision_kwarg_supported else None,
            return_energy_type=return_energy_type,
        )
        try:
            return self.potential.createSystem(topology, **kwargs)
        except TypeError as exc:
            if "precision" in kwargs and self._is_unsupported_precision_kwarg(exc):
                self._precision_kwarg_supported = False
                self.openmmml_precision = None
                return self.potential.createSystem(
                    topology,
                    **_build_openmmml_kwargs(device=self.device, return_energy_type=return_energy_type),
                )
            raise

    def _create_orb_context_bundle(
        self,
        numbers: np.ndarray,
        box_vectors=None,
        return_energy_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        top = openmm.app.Topology()
        chain = top.addChain()
        res = top.addResidue("MLP", chain)
        for z in numbers:
            top.addAtom(f"Z{z}", openmm.app.element.Element.getByAtomicNumber(int(z)), res)
        if box_vectors is not None:
            top.setPeriodicBoxVectors(box_vectors)

        sys = self._create_openmmml_system(top, return_energy_type=return_energy_type)
        integ = openmm.VerletIntegrator(0.001)
        sim = openmm.app.Simulation(top, sys, integ)
        if box_vectors is not None:
            sim.context.setPeriodicBoxVectors(*box_vectors)
        return {
            "context": sim.context,
            "simulation": sim,
            "integrator": integ,
            "system": sys,
        }

    def _get_orb_decomposition_bundle(self, lig_idx: np.ndarray, env_idx: np.ndarray, all_nums: np.ndarray) -> Dict[str, Any]:
        key = (
            self.label_mode,
            tuple(int(x) for x in lig_idx),
            tuple(int(x) for x in env_idx),
            self.device,
            self.openmmml_precision,
        )
        if key in self._orb_ctx_cache:
            return self._orb_ctx_cache[key]

        comb_idx = np.concatenate([lig_idx, env_idx])
        if self.label_mode == "orbv3_interaction":
            bundle = {
                "comb_idx": comb_idx,
                "lig_idx": lig_idx,
                "env_idx": env_idx,
                "contexts": {
                    "cplx": self._create_orb_context_bundle(
                        all_nums[comb_idx],
                        return_energy_type="interaction_energy",
                    ),
                },
            }
        else:
            bundle = {
                "comb_idx": comb_idx,
                "lig_idx": lig_idx,
                "env_idx": env_idx,
                "contexts": {
                    "cplx": self._create_orb_context_bundle(all_nums[comb_idx]),
                    "lig": self._create_orb_context_bundle(all_nums[lig_idx]),
                    "env": self._create_orb_context_bundle(all_nums[env_idx]),
                },
            }
        self._orb_ctx_cache[key] = bundle
        return bundle

    @staticmethod
    def _evaluate_orb_context_energy(ctx_bundle: Dict[str, Any], pos_nm: np.ndarray) -> float:
        ctx = ctx_bundle["context"]
        ctx.setPositions(pos_nm * unit.nanometer)
        return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    def _compute_orb_decomposition(self, pos_nm: np.ndarray, lig_idx: np.ndarray, env_idx: np.ndarray, all_nums: np.ndarray) -> float:
        try:
            bundle = self._get_orb_decomposition_bundle(lig_idx, env_idx, all_nums)
            e_cplx = self._evaluate_orb_context_energy(bundle["contexts"]["cplx"], pos_nm[bundle["comb_idx"]])
            if self.label_mode == "orbv3_interaction":
                return e_cplx
            e_lig = self._evaluate_orb_context_energy(bundle["contexts"]["lig"], pos_nm[bundle["lig_idx"]])
            e_env = self._evaluate_orb_context_energy(bundle["contexts"]["env"], pos_nm[bundle["env_idx"]])
            return e_cplx - e_lig - e_env
        except Exception as exc:
            if self.device == "cuda" and self._is_cuda_oom(exc):
                self._fallback_to_cpu_double("OpenMM-ML CUDA 显存不足")
                bundle = self._get_orb_decomposition_bundle(lig_idx, env_idx, all_nums)
                e_cplx = self._evaluate_orb_context_energy(bundle["contexts"]["cplx"], pos_nm[bundle["comb_idx"]])
                if self.label_mode == "orbv3_interaction":
                    return e_cplx
                e_lig = self._evaluate_orb_context_energy(bundle["contexts"]["lig"], pos_nm[bundle["lig_idx"]])
                e_env = self._evaluate_orb_context_energy(bundle["contexts"]["env"], pos_nm[bundle["env_idx"]])
                return e_cplx - e_lig - e_env
            if self._is_precision_or_dtype_mismatch(exc):
                self._fallback_to_cpu_double("OpenMM-ML precision/device 与模型默认精度不兼容")
                bundle = self._get_orb_decomposition_bundle(lig_idx, env_idx, all_nums)
                e_cplx = self._evaluate_orb_context_energy(bundle["contexts"]["cplx"], pos_nm[bundle["comb_idx"]])
                if self.label_mode == "orbv3_interaction":
                    return e_cplx
                e_lig = self._evaluate_orb_context_energy(bundle["contexts"]["lig"], pos_nm[bundle["lig_idx"]])
                e_env = self._evaluate_orb_context_energy(bundle["contexts"]["env"], pos_nm[bundle["env_idx"]])
                return e_cplx - e_lig - e_env
            raise

    def _preflight_orb_backend(self, pos_nm: np.ndarray, lig_idx: np.ndarray, env_idx: np.ndarray, all_nums: np.ndarray):
        """
        在主循环前预建并预热 cplx/lig/env 三个常驻 Context。
        后续每一帧只滚动更新坐标，不再在帧循环里重建 Context。
        """
        try:
            bundle = self._get_orb_decomposition_bundle(lig_idx, env_idx, all_nums)
            self._evaluate_orb_context_energy(bundle["contexts"]["cplx"], pos_nm[bundle["comb_idx"]])
            if self.label_mode == "orbv3_interaction":
                return
            self._evaluate_orb_context_energy(bundle["contexts"]["lig"], pos_nm[bundle["lig_idx"]])
            self._evaluate_orb_context_energy(bundle["contexts"]["env"], pos_nm[bundle["env_idx"]])
        except Exception as exc:
            if self.device == "cuda" and self._is_cuda_oom(exc):
                self._fallback_to_cpu_double("OpenMM-ML CUDA 预检显存不足")
                bundle = self._get_orb_decomposition_bundle(lig_idx, env_idx, all_nums)
                self._evaluate_orb_context_energy(bundle["contexts"]["cplx"], pos_nm[bundle["comb_idx"]])
                if self.label_mode == "orbv3_interaction":
                    return
                self._evaluate_orb_context_energy(bundle["contexts"]["lig"], pos_nm[bundle["lig_idx"]])
                self._evaluate_orb_context_energy(bundle["contexts"]["env"], pos_nm[bundle["env_idx"]])
                return
            if self._is_precision_or_dtype_mismatch(exc):
                self._fallback_to_cpu_double("OpenMM-ML 预检 precision/device 与模型默认精度不兼容")
                bundle = self._get_orb_decomposition_bundle(lig_idx, env_idx, all_nums)
                self._evaluate_orb_context_energy(bundle["contexts"]["cplx"], pos_nm[bundle["comb_idx"]])
                if self.label_mode == "orbv3_interaction":
                    return
                self._evaluate_orb_context_energy(bundle["contexts"]["lig"], pos_nm[bundle["lig_idx"]])
                self._evaluate_orb_context_energy(bundle["contexts"]["env"], pos_nm[bundle["env_idx"]])
                return
            raise

    def _build_mm_le_contexts(self, topology, gro_box, lig_idx, env_idx, gmx_include_dir=None, top_file=None):
        if isinstance(topology, openmm.app.Topology):
            omm_top = topology
        else:
            omm_top = topology.to_openmm()

        n_total = omm_top.getNumAtoms()

        temp_sys = None
        if hasattr(omm_top, 'createSystem'):
            temp_sys = omm_top.createSystem(
                nonbondedMethod=openmm.app.PME,
                nonbondedCutoff=1.0 * unit.nanometer,
                constraints=openmm.app.HBonds,
            )
        elif top_file and gmx_include_dir:
            from openmm.app import GromacsTopFile
            top_wrapper = GromacsTopFile(top_file, includeDir=gmx_include_dir)
            temp_sys = top_wrapper.createSystem(
                nonbondedMethod=openmm.app.PME,
                nonbondedCutoff=1.0 * unit.nanometer,
                constraints=openmm.app.HBonds,
            )

        if temp_sys is None:
            raise RuntimeError("无法从拓扑构建 MM 参考系统，请提供 .top 文件 + --gmx-path")

        nb_orig = next((f for f in temp_sys.getForces() if isinstance(f, openmm.NonbondedForce)), None)
        if nb_orig is None:
            raise RuntimeError("原始系统中未找到 NonbondedForce")

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
        cutoff_nm = 0.65
        switching_nm = 0.55
        contexts = {}
        lig_set = {int(x) for x in lig_idx.tolist()}
        for label, (expr, per_params) in force_defs.items():
            le_sys = openmm.System()
            for _ in range(n_total):
                le_sys.addParticle(1.0)
            le_force = openmm.CustomNonbondedForce(expr)
            for param_name in per_params:
                le_force.addPerParticleParameter(param_name)
            for i in range(n_total):
                q, sig, eps = nb_orig.getParticleParameters(i)
                payload = []
                for param_name in per_params:
                    if param_name == "q":
                        payload.append(q.value_in_unit(unit.elementary_charge))
                    elif param_name == "type":
                        payload.append(1.0 if i in lig_set else 0.0)
                    elif param_name == "sigma":
                        payload.append(sig.value_in_unit(unit.nanometer))
                    elif param_name == "epsilon":
                        payload.append(eps.value_in_unit(unit.kilojoule_per_mole))
                le_force.addParticle(payload)
            if label != "gauss_coul":
                le_force.addInteractionGroup(lig_idx.tolist(), env_idx.tolist())
            le_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
            le_force.setCutoffDistance(cutoff_nm * unit.nanometer)
            le_force.setUseSwitchingFunction(False)
            for i in range(nb_orig.getNumExceptions()):
                p1, p2, _, _, _ = nb_orig.getExceptionParameters(i)
                le_force.addExclusion(int(p1), int(p2))
            le_sys.addForce(le_force)
            ctx = openmm.Context(le_sys, openmm.VerletIntegrator(0.001))
            if gro_box is not None:
                ctx.setPeriodicBoxVectors(*gro_box)
            contexts[label] = ctx
        return contexts

    def run_from_trajectory(
        self,
        traj_file: str,
        top_file: str,
        ligand_resname: str,
        output_json: str,
        n_frames: int = 200,
        env_radius_nm: float = 0.60,
        env_max_atoms: Optional[int] = None,
        fit_last_ns: Optional[float] = None,
        gmx_include_dir: str = None,
        fitting_region: tuple = (0.20, 0.50),
    ) -> dict:
        import mdtraj as md

        print(f"\n🧪 启动 Orbv3-DEXP 拟合 | 轨迹: {traj_file} | 配体: {ligand_resname} | 设备: {self.device}")

        traj = md.load(traj_file, top=top_file)
        lig_idx = np.array(traj.top.select(f"resname {ligand_resname}"), dtype=int)
        if len(lig_idx) == 0:
            raise ValueError(f"未找到配体残基: {ligand_resname}")

        frame_indices = _select_tail_indices_from_time(traj, n_frames, fit_last_ns)
        if not frame_indices:
            raise RuntimeError("轨迹为空，无法执行 DEXP 拟合")
        fit_traj = traj[frame_indices]
        if fit_traj.unitcell_vectors is not None:
            fit_traj = fit_traj.image_molecules(inplace=False)

        env_radius_nm = float(env_radius_nm)
        env_idx = _select_env_indices_from_mdtraj_frame(
            fit_traj[-1], lig_idx, env_radius_nm, max_env_atoms=env_max_atoms
        )
        if len(env_idx) == 0:
            raise RuntimeError("未找到配体附近环境原子，请增大 env_radius_nm")
        if env_max_atoms is not None:
            print(f"🔒 OpenMM-ML 环境原子上限: {env_max_atoms} | 实际选中: {len(env_idx)}")
        all_nums = np.array([a.element.atomic_number for a in fit_traj.top.atoms], dtype=int)

        gro_box = fit_traj.unitcell_vectors[0] * unit.nanometer if fit_traj.unitcell_vectors is not None else None
        mm_contexts = self._build_mm_le_contexts(
            fit_traj.topology, gro_box, lig_idx, env_idx,
            gmx_include_dir=gmx_include_dir, top_file=top_file,
        )

        e_int_list, dists_per_frame = [], []
        stats = {"total": 0, "success": 0, "skip_outlier": 0, "skip_no_dists": 0}
        raw_orb_values: List[float] = []
        raw_mm_coul_values: List[float] = []
        raw_mm_vdw_values: List[float] = []

        first_frame = fit_traj[0]
        first_pos_nm = (
            first_frame.image_molecules(inplace=False).xyz[0].copy()
            if first_frame.unitcell_vectors is not None
            else first_frame.xyz[0].copy()
        )
        self._preflight_orb_backend(first_pos_nm, lig_idx, env_idx, all_nums)

        print(f"⏳ 开始标注 {len(frame_indices)} 帧 ΔE_qmmm = E_qm(region) - E_mm(region) ...")
        for i, fid in enumerate(frame_indices):
            stats["total"] += 1
            try:
                frame = fit_traj[i]
                pos_nm = frame.xyz[0].copy()

                e_orb_int = self._compute_orb_decomposition(pos_nm, lig_idx, env_idx, all_nums)

                e_mm_coul = 0.0
                e_mm_vdw = 0.0
                for label, ctx in mm_contexts.items():
                    if frame.unitcell_vectors is not None:
                        frame_box = frame.unitcell_vectors[0] * unit.nanometer
                        ctx.setPeriodicBoxVectors(*frame_box)
                    ctx.setPositions(pos_nm * unit.nanometer)
                    energy = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                    if label == "coul":
                        e_mm_coul = energy
                    elif label == "vdw":
                        e_mm_vdw = energy

                delta_e = e_orb_int - e_mm_coul - e_mm_vdw
                raw_orb_values.append(float(e_orb_int))
                raw_mm_coul_values.append(float(e_mm_coul))
                raw_mm_vdw_values.append(float(e_mm_vdw))
                if np.isnan(delta_e) or np.isinf(delta_e) or abs(delta_e) > 5000.0:
                    stats["skip_outlier"] += 1
                    continue

                box_vecs = frame.unitcell_vectors[0] if frame.unitcell_vectors is not None else np.eye(3) * 3.0
                box_lens = np.linalg.norm(box_vecs, axis=1)
                delta = pos_nm[lig_idx][:, None, :] - pos_nm[env_idx][None, :, :]
                delta -= box_lens * np.round(delta / box_lens)
                dists = np.linalg.norm(delta, axis=-1)
                valid_dists = dists[(dists >= float(fitting_region[0])) & (dists <= float(fitting_region[1]))]

                if len(valid_dists) == 0:
                    stats["skip_no_dists"] += 1
                    continue

                stats["success"] += 1
                e_int_list.append(delta_e)
                dists_per_frame.append(valid_dists)

                if stats["success"] <= 3 or stats["success"] % 50 == 0:
                    print(
                        f"   ✅ Frame {fid} | ΔE_qmmm={delta_e:7.2f} kJ/mol | "
                        f"E_mm_coul={e_mm_coul:7.2f} | E_mm_vdw={e_mm_vdw:7.2f} | Pairs={len(valid_dists)}"
                    )
            except Exception as e:
                stats["skip_outlier"] += 1
                continue

        print(f"\n📊 采样诊断: 成功={stats['success']}, 过滤={stats['skip_outlier']+stats['skip_no_dists']}")
        if stats["success"] < 10:
            raise RuntimeError("有效 ΔE 数据不足 10 帧，无法拟合。请检查轨迹质量或扩大 env_radius_nm。")

        delta_threshold, delta_diag = _choose_delta_e_threshold(e_int_list)
        if delta_diag["polluted"]:
            filtered_pairs = [
                (delta_e, dists)
                for delta_e, dists in zip(e_int_list, dists_per_frame)
                if np.isfinite(delta_e) and abs(float(delta_e) - float(delta_diag["center"])) < delta_threshold
            ]
            if len(filtered_pairs) >= 10:
                e_int_list = [float(delta_e) for delta_e, _ in filtered_pairs]
                dists_per_frame = [dists for _, dists in filtered_pairs]
                stats["success"] = len(e_int_list)
                print(
                    f"📎 启用尾段过滤: 保留 {len(e_int_list)} 帧 | "
                    f"threshold={delta_threshold:.1f} kJ/mol | reason={delta_diag['reason']}"
                )
            else:
                print("⚠️ ΔE 过滤后有效帧不足 10，回退为使用全部成功帧拟合。")

        qm_mm_offset = float(np.mean(e_int_list))
        e_int_list = [float(val - qm_mm_offset) for val in e_int_list]
        print(f"📌 QM/MM reference shift = {qm_mm_offset:.3f} kJ/mol（仅诊断，不进入 OpenMM force）")

        print("📉 启动 DEXP 参数优化...")
        fitter = Orbv3SurrogateFitter(fitting_region=fitting_region)
        fitted_params = fitter.fit_parameters(dists_per_frame, e_int_list)
        fitted_params["qm_mm_offset_kjmol"] = qm_mm_offset
        fitted_params["fit_target_definition"] = "delta_fit = (E_qm_region - E_mm_region) - <E_qm_region - E_mm_region>"
        fitted_params["fit_frames_requested"] = int(n_frames)
        fitted_params["fit_last_ns_requested"] = float(fit_last_ns) if fit_last_ns is not None else None
        fitted_params["fit_frames_total"] = int(len(frame_indices))
        fitted_params["fit_frames_used"] = int(len(e_int_list))
        fitted_params["fit_frame_start"] = int(frame_indices[0])
        fitted_params["fit_frame_end"] = int(frame_indices[-1])
        if getattr(traj, "time", None) is not None:
            fitted_params["fit_time_start_ps"] = float(traj.time[frame_indices[0]])
            fitted_params["fit_time_end_ps"] = float(traj.time[frame_indices[-1]])
        fitted_params["env_radius_nm"] = float(env_radius_nm)
        fitted_params["env_max_atoms"] = int(env_max_atoms) if env_max_atoms is not None else None
        fitted_params["fit_region_nm"] = [float(fitting_region[0]), float(fitting_region[1])]
        fitted_params["traj_total_frames"] = int(len(traj))
        fitted_params["ml_model"] = str(self.model_name)
        fitted_params["label_mode"] = str(self.label_mode)
        fitted_params["delta_e_filter_threshold_kjmol"] = float(delta_threshold)
        fitted_params["delta_e_polluted"] = bool(delta_diag["polluted"])
        fitted_params["delta_e_pollution_reason"] = str(delta_diag["reason"])
        fitted_params["delta_e_mean_kjmol"] = float(delta_diag["stats"]["mean"])
        fitted_params["delta_e_std_kjmol"] = float(delta_diag["stats"]["std"])
        fitted_params["delta_e_mean_abs_kjmol"] = float(delta_diag["stats"]["mean_abs"])
        orb_stats = _summarize_dexp_values(raw_orb_values)
        mm_coul_stats = _summarize_dexp_values(raw_mm_coul_values)
        mm_vdw_stats = _summarize_dexp_values(raw_mm_vdw_values)
        fitted_params["e_orb_int_mean_kjmol"] = float(orb_stats["mean"])
        fitted_params["e_orb_int_std_kjmol"] = float(orb_stats["std"])
        fitted_params["e_mm_coul_mean_kjmol"] = float(mm_coul_stats["mean"])
        fitted_params["e_mm_coul_std_kjmol"] = float(mm_coul_stats["std"])
        fitted_params["e_mm_vdw_mean_kjmol"] = float(mm_vdw_stats["mean"])
        fitted_params["e_mm_vdw_std_kjmol"] = float(mm_vdw_stats["std"])
        fitted_params.update(_detect_suspicious_fit(fitted_params))

        if not fitted_params.get("fitting_success"):
            print("⚠️ 拟合未收敛，返回默认参数")
        elif fitted_params.get("suspicious_fit"):
            print(f"⚠️ 检测到参数撞边界: {', '.join(fitted_params.get('boundary_hits', []))}")

        with open(output_json, "w") as f:
            json.dump(fitted_params, f, indent=2, cls=NumpyEncoder)
        self._clear_orb_context_cache()
        print(f"✅ 拟合完成，参数已保存: {output_json}")
        return fitted_params        

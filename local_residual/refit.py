"""EXP-033 §5 P1：把「换配体重训 B_φ」压成一次闭式求解。

`docs/EXP-033_LOCAL_RESIDUAL_REFIT_2026-09-10.md` §5 P1 的五条，逐条落在这里：

1. 闭式拟合。目标函数就是现有的 `bidirectional_gap_variance_loss`——它对 `B` 二次，
   而线性 `B` 对参数二次 ⟹ `θ* = −(M + λΩ)⁻¹ b`。正则是 **r 方向的二阶平滑先验**，
   不是纯岭：§2.3C 第 3 条实测纯岭会让三折系数相关掉到 −0.33、幅度到 1e11 kT
   互相抵消。λ 用**内层** CV 选，不看测试折。不 import torch、不跑 SGD。
2. 口径对账。标签**不读** `energies.npy`，用逐态软核 probe 现算 —— 于是 §2.2 那个
   「LRC 算没算进去、单位是 kJ/mol 还是 reduced」的风险根本不存在（前科见
   `docs/STAGE2_SOLVENT_LEG_ERROR_BUDGET.md`）。
3. 容量常量按配体尺寸推导（`_derived_capacities`），并与**实测**用量取大，
   替掉 `softlift.py` 里按 41 原子写死的 320/2048/80。
4. 词表从 teacher z-table ∩ 体系推（给了 teacher 才有交集，否则退回体系元素并在
   报告里写明）——替掉 `softlift_dataset.py` 的 `unique(all_topology_atomic_numbers)`。
5. 线性 `B` 无界（§2.3B 实测量程 ±28~40 kT，而 `max_abs_basis` 只有 12 kT）。
   这里取 §5 item 5 的**两条一起**：先减掉常数（逐态常数被 f_k 吸收，对 ΔG 无影响），
   再把 tanh 留成安全界，并把 tanh 实际咬合的帧比例写进报告、超限 fail-closed。

## 帧源

`pre_equilibration.dcd`，有多少帧用多少。P1 item 1 原文写的是「那次 run 自己的
`energies.npy` / `sampling_states.npy` / `residual_basis.npy` + DCD」，那是因为当时
有 rerun1/2/3 三条平行实验、LORO 可以按**独立 run** 分折。**`B_φ` 在 outer、不进
哈密顿量**（physical target 永远不含残差），所以换成预平衡帧不改变 ΔG 的正确性，
只改变这个采样增强项好不好用。于是分折在这里只为 item 1 的内层 CV 服务，
按时间切段即可——而 LORO held-out 本来就**不是验收判据**（§4）。

## 这份产物不是验收

离线 gap-variance 改善只是「值得上机的信号」。真验收是 EXP-027 U3 口径：
window-0 utility + ΔG 一致性，且**两臂各自独立标定并冻结自己的 f_k**（U4 就是栽在
候选臂复用 baseline 的 f_k，被封为 `INVALID_FOR_PROMOTION`）。那是 EXP-033 P2。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "RefitError",
    "affine_rho_absolute_error_floor",
    "affine_rho_is_exact",
    "closed_form_refit",
    "a_k_schedule",
    "frame_basis_features",
    "solve_closed_form",
]


class RefitError(RuntimeError):
    pass


# `rho` 被构造成**恰好仿射**时用的偏置。SiLU(z) = z·sigmoid(z)，而 sigmoid(z) 在
# 1 − e^(−z) 舍入到 1.0 时**逐比特**等于 1 ⟹ SiLU(z) == z。float64 要 e^(−z) < eps/2
# ≈ 1.1e-16，即 z > 36.7；留到 45 以上才用。下面用 L=60 且把 |s·q| 夹在 15 以内，
# 最小预激活 45、次层 105，float64/float32 两边都精确。
_RHO_AFFINE_BIAS = 60.0
_RHO_AFFINE_MAX_Q = 15.0
_RHO_HIDDEN = 16


def _log_to(log: Callable[[str], None] | None) -> Callable[[str], None]:
    return log if log is not None else (lambda _message: None)


def a_k_schedule(lambdas_vdw: Sequence[float]) -> tuple[list[float], list[float]]:
    """`A_k = sin^2(pi*lambda)`，端点严格 0。

    必须与运行时 `outer_lambda_neural_basis.OuterLambdaController.envelope` 逐字同一条
    ——训练用的系数与运行时不是同一条，学的就不是同一个量。
    `tests/test_local_residual_refit.py` 钉住这一点。
    """

    values = [float(value) for value in lambdas_vdw]
    a_k = [
        0.0 if (lam <= 0.0 or lam >= 1.0) else math.sin(math.pi * lam) ** 2
        for lam in values
    ]
    delta = [a_k[index + 1] - a_k[index] for index in range(len(a_k) - 1)]
    return delta, a_k


def _derived_capacities(n_ligand_atoms: int) -> dict[str, int]:
    """容量按配体尺寸缩放；41 原子时与出厂那组 (320/2048/80) 逐值相同。

    `max_neighbors_per_ligand` 是**每个配体原子**的邻居上限，不随配体大小变。
    """

    from .softlift import R1_REFERENCE_LIGAND_ATOM_COUNT

    n = int(n_ligand_atoms)
    if n <= 0:
        raise RefitError("n_ligand_atoms 必须为正")
    scale = max(1.0, n / float(R1_REFERENCE_LIGAND_ATOM_COUNT))
    return {
        "max_environment_atoms": int(math.ceil(320 * scale)),
        "max_edges": int(math.ceil(2048 * scale)),
        "max_neighbors_per_ligand": 80,
    }


def _quintic_c2(distance, inner: float, outer: float):
    """与 `softlift._quintic_c2` 同一条解析式的 numpy 版。"""

    import numpy as np

    x = (distance - inner) / (outer - inner)
    transition = 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5
    return np.where(
        distance <= inner, 1.0, np.where(distance >= outer, 0.0, transition)
    )


def _radial_basis(distance, centers, width: float):
    """`G_p(r) = exp(-0.5 * ((r - centers[p]) / width)^2)`，与 R1 模型同式。"""

    import numpy as np

    diff = np.asarray(distance, dtype=np.float64)[:, None] - centers[None, :]
    return np.exp(-0.5 * (diff / width) ** 2)


def frame_basis_features(
    *,
    positions_angstrom,
    box_angstrom,
    ligand_indices,
    environment_indices,
    ligand_type_index,
    environment_type_index,
    n_types: int,
    radial_centers,
    radial_width: float,
    inner_cutoff_angstrom: float,
    outer_cutoff_angstrom: float,
):
    """一帧的特征 `φ` 与实测容量。

    `raw = Σ_i q_i = θ · φ`，其中

        φ[t1, t2, p] = Σ_{边 e 的类型对是 (t1,t2)} envelope(r_e) · G_p(r_e)

    这一步就是把 R1 前向里 `pair_weight` 的**线性**部分提出来：ρ 被构造成恰好仿射
    （见 `_affine_rho_state`），斜率折进 `pair_weight`，所以 `raw` 对 θ 线性。
    """

    import numpy as np

    from .softlift_dataset import _numpy_cross_edges

    edge_ligand, edge_env, _displacement, distance = _numpy_cross_edges(
        positions_angstrom,
        box_angstrom,
        ligand_indices,
        environment_indices,
        outer_cutoff_angstrom,
    )
    features = np.zeros((n_types, n_types, len(radial_centers)), dtype=np.float64)
    realized = {"edges": int(distance.size), "environment_atoms": 0, "neighbors": 0}
    if distance.size:
        envelope = _quintic_c2(distance, inner_cutoff_angstrom, outer_cutoff_angstrom)
        radial = _radial_basis(distance, radial_centers, radial_width)
        contribution = envelope[:, None] * radial
        ligand_order = np.asarray(ligand_indices, dtype=np.int64)
        local = np.searchsorted(ligand_order, edge_ligand)
        t1 = np.asarray(ligand_type_index, dtype=np.int64)[local]
        t2 = np.asarray(environment_type_index, dtype=np.int64)[edge_env]
        flat = (t1 * n_types + t2).astype(np.int64)
        # 按类型对累加；np.add.at 对重复下标是正确的（+= 会丢掉重复）。
        accumulator = features.reshape(n_types * n_types, -1)
        np.add.at(accumulator, flat, contribution)
        realized["environment_atoms"] = int(np.unique(edge_env).size)
        realized["neighbors"] = int(np.bincount(local, minlength=len(ligand_order)).max())
    return features.reshape(-1), realized


def _normalized_weights(log_importance_column):
    """一列 `log_importance_unnormalized` → 归一权重（log-sum-exp 稳定化）。"""

    import numpy as np

    shifted = log_importance_column - np.max(log_importance_column)
    weights = np.exp(shifted)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise RefitError("重要性权重的总质量非有限或非正")
    return weights / total


def _smoothness_prior(n_types: int, n_radial: int):
    """r 方向的二阶差分先验 `Ω = D₂ᵀD₂`，逐 (t1,t2) 块对角。

    §2.3C 第 3 条：纯岭下三折系数相关 −0.33、幅度 1e11 kT 相互抵消（预测仍泛化，
    但参数完全不可辨识）。平滑先验把相关拉回 +0.65，所以正则必须是这一条。
    另叠一个极小的岭项，只为数值可逆，不承担统计作用。
    """

    import numpy as np

    if n_radial < 3:
        return np.eye(n_types * n_types * n_radial)
    d2 = np.zeros((n_radial - 2, n_radial))
    for row in range(n_radial - 2):
        d2[row, row] = 1.0
        d2[row, row + 1] = -2.0
        d2[row, row + 2] = 1.0
    block = d2.T @ d2 + 1.0e-8 * np.eye(n_radial)
    return np.kron(np.eye(n_types * n_types), block)


def _quadratic_form(features, gaps, log_importance, delta_a, frame_subset):
    """把 loss 在给定帧集合上攒成 `(M, b, c)`：`L(θ) = θᵀMθ + 2bᵀθ + c`。

    与 `local_residual/loss.py` 的 `bidirectional_gap_variance_loss` 同一条定义：
    每条相邻边的修正 gap 是 `adjacent_gap_reduced + delta_A * B`，边 loss 是它在
    **两个**目标态下加权方差的各一半，总 loss 是所有 (分区, 边) 值的算术平均。
    """

    import numpy as np

    subset = np.asarray(frame_subset, dtype=np.int64)
    phi = np.asarray(features, dtype=np.float64)[subset]
    n_edges = gaps.shape[1]
    dim = phi.shape[1]
    M = np.zeros((dim, dim), dtype=np.float64)
    b = np.zeros(dim, dtype=np.float64)
    c = 0.0
    for edge in range(n_edges):
        g = np.asarray(gaps, dtype=np.float64)[subset, edge]
        scale = float(delta_a[edge])
        for state in (edge, edge + 1):
            w = _normalized_weights(np.asarray(log_importance)[subset, state])
            g_centered = g - float(w @ g)
            phi_centered = phi - (w @ phi)[None, :]
            weighted = phi_centered * w[:, None]
            # 0.5（两个目标态各一半）/ n_edges（对边取算术平均）
            factor = 0.5 / n_edges
            M += factor * (scale**2) * (phi_centered.T @ weighted)
            b += factor * scale * (weighted.T @ g_centered)
            c += factor * float(w @ (g_centered**2))
    return M, b, c


def _evaluate_loss(theta, features, gaps, log_importance, delta_a, frame_subset):
    import numpy as np

    M, b, c = _quadratic_form(features, gaps, log_importance, delta_a, frame_subset)
    theta = np.asarray(theta, dtype=np.float64)
    return float(theta @ M @ theta + 2.0 * b @ theta + c)


def solve_closed_form(
    *,
    features,
    adjacent_gap_reduced,
    log_importance_unnormalized,
    delta_a,
    fold_index,
    ridge_grid=None,
    n_types: int | None = None,
    n_radial: int = 16,
    log: Callable[[str], None] | None = None,
):
    """`θ* = −(M + λΩ)⁻¹ b`，λ 由**内层** CV 选（不看测试折）。

    返回 `(theta, report)`。`report` 里记每折的 held-out loss 与选中的 λ ——
    那只是过拟合的内部诊断，**不是验收判据**（EXP-033 §4）。
    """

    import numpy as np

    _log = _log_to(log)
    phi = np.asarray(features, dtype=np.float64)
    n_frames, dim = phi.shape
    folds = np.asarray(fold_index, dtype=np.int64)
    unique_folds = sorted(set(folds.tolist()))
    if len(unique_folds) < 2:
        raise RefitError("内层 CV 至少要两折")
    if ridge_grid is None:
        ridge_grid = [10.0**power for power in range(-8, 5)]
    omega = (
        _smoothness_prior(int(n_types), int(n_radial))
        if n_types is not None
        else _smoothness_prior_for_dim(dim)
    )

    scores: list[dict[str, Any]] = []
    for candidate in ridge_grid:
        held_out = []
        for fold in unique_folds:
            train = np.flatnonzero(folds != fold)
            test = np.flatnonzero(folds == fold)
            if train.size == 0 or test.size == 0:
                raise RefitError("内层 CV 出现空折")
            M, b, _c = _quadratic_form(
                phi, adjacent_gap_reduced, log_importance_unnormalized, delta_a, train
            )
            theta = _solve(M, b, omega, candidate)
            held_out.append(
                _evaluate_loss(
                    theta, phi, adjacent_gap_reduced,
                    log_importance_unnormalized, delta_a, test,
                )
            )
        scores.append({"ridge": float(candidate), "mean_held_out_loss": float(np.mean(held_out)),
                       "per_fold_held_out_loss": [float(v) for v in held_out]})
    best = min(scores, key=lambda row: row["mean_held_out_loss"])
    _log(f"  [refit] 内层 CV 选中 λ = {best['ridge']:.3g}")

    M, b, c = _quadratic_form(
        phi, adjacent_gap_reduced, log_importance_unnormalized, delta_a,
        np.arange(n_frames),
    )
    theta = _solve(M, b, omega, best["ridge"])
    baseline_loss = float(c)
    fitted_loss = float(theta @ M @ theta + 2.0 * b @ theta + c)
    report = {
        "ridge_selected": best["ridge"],
        "ridge_grid_scores": scores,
        "loss_without_basis": baseline_loss,
        "loss_with_basis": fitted_loss,
        "relative_improvement": (
            (baseline_loss - fitted_loss) / baseline_loss if baseline_loss > 0 else 0.0
        ),
        "n_frames": int(n_frames),
        "n_parameters": int(dim),
    }
    return theta, report


def _smoothness_prior_for_dim(dim: int):
    """维度 = n_types² × n_radial，n_radial 固定 16（R1 架构常量）。"""

    import numpy as np

    n_radial = 16
    if dim % n_radial:
        raise RefitError(f"特征维度 {dim} 不是 n_radial=16 的整数倍")
    n_types_sq = dim // n_radial
    n_types = int(round(math.sqrt(n_types_sq)))
    if n_types * n_types != n_types_sq:
        raise RefitError(f"特征维度 {dim} 推不出整数类型数")
    return _smoothness_prior(n_types, n_radial)


def _solve(M, b, omega, ridge: float):
    import numpy as np

    system = M + float(ridge) * omega
    try:
        return -np.linalg.solve(system, b)
    except np.linalg.LinAlgError as exc:
        raise RefitError(f"闭式解的法方程奇异（λ={ridge:.3g}）：{exc}") from exc


# =============================================================================
# 恰好仿射的 rho —— 闭式解只给 pair_weight，rho 必须填成不改变线性性的形状
# =============================================================================
def _affine_rho_state(n_types: int, slope_scale: float) -> dict[str, Any]:
    """构造一组 rho 权重，使 `rho(q) − rho(0)` 等于 `q`。

    payload 的 rho 形状是硬写的
    `Linear(1,16) → SiLU → Linear(16,16) → SiLU → Linear(16,1)`
    （`openmm_plugin.load_r1_payload`），而闭式解只解得出 `pair_weight`。
    利用 `SiLU(z) = z·sigmoid(z)` 在 `sigmoid(z)` 舍入到 `1.0` 时**精确**退化成恒等：

        h0 = [s·q + L, L, …, L]            (w0 = s·e₀, b0 = L)
        z2 = [s·q + L, 0, …, 0]            (w2 行0 = e₀ᵀ, 其余 0, b2 = 0)
        rho(q) = g·(s·q + L)               (w4 = g·e₀ᵀ, b4 = 0)
        rho(q) − rho(0) = g·s·q

    取 `g·s = 1`，斜率整体折进 `pair_weight`。`s` 由调用方按实测 `|q|` 上界定，
    保证最小预激活 ≥ 45 —— float64 要 `e^(−z) < eps/2`，即 `z > 36.7`。
    `b2` 取 0（而不是再叠一个 L）是为了只付**一次**偏置：次层预激活
    `h0[0] = s·q + L ≥ 45` 本来就够大，叠第二个 L 只会白白多丢一半精度。

    ⚠️ **不是逐比特恒等。** SiLU 那一步是精确的，但 `(s·q + L) − L` 这个加了再减
    会吃掉低位：绝对误差量级 `eps(L)/s ≈ 7.1e-15/s`（见
    `affine_rho_absolute_error_floor`）。对 `b_max = 10` 的工作点而言是 1e-15 量级的
    相对误差，物理上无意义，但**别把它写成"逐比特"**。
    """

    import numpy as np

    s = float(slope_scale)
    if not (s > 0.0) or not math.isfinite(s):
        raise RefitError(f"rho 仿射斜率必须是正有限数，收到 {slope_scale!r}")
    g = 1.0 / s
    state: dict[str, Any] = {}
    for t in range(n_types):
        w0 = np.zeros((_RHO_HIDDEN, 1), dtype=np.float64)
        w0[0, 0] = s
        w2 = np.zeros((_RHO_HIDDEN, _RHO_HIDDEN), dtype=np.float64)
        w2[0, 0] = 1.0
        w4 = np.zeros((1, _RHO_HIDDEN), dtype=np.float64)
        w4[0, 0] = g
        state[f"rho.{t}.0.weight"] = w0
        state[f"rho.{t}.0.bias"] = np.full(_RHO_HIDDEN, _RHO_AFFINE_BIAS)
        state[f"rho.{t}.2.weight"] = w2
        state[f"rho.{t}.2.bias"] = np.zeros(_RHO_HIDDEN, dtype=np.float64)
        state[f"rho.{t}.4.weight"] = w4
        state[f"rho.{t}.4.bias"] = np.zeros(1, dtype=np.float64)
    return state


def affine_rho_is_exact(q_values, slope_scale: float) -> bool:
    """在给定 `q` 量程上，SiLU 那一步是否仍然精确退化成恒等。

    判据是最小预激活：`L − max|s·q| ≥ 45`（float64 的 `e^(−45) = 2.9e-20`，远小于
    `eps/2 = 1.1e-16`，`sigmoid` 舍入到 `1.0`）。**这只管 SiLU**，偏置加减的低位
    损失是另一回事，见 `affine_rho_absolute_error_floor`。
    """

    import numpy as np

    scaled = float(slope_scale) * np.abs(np.asarray(q_values, dtype=np.float64)).max()
    return bool(scaled <= _RHO_AFFINE_MAX_Q and _RHO_AFFINE_BIAS - scaled >= 45.0)


def affine_rho_absolute_error_floor(slope_scale: float) -> float:
    """`rho(q) − rho(0)` 相对 `q` 的绝对误差下限 —— 偏置加了再减吃掉的低位。

    `s·q + L` 的 ULP 是 `eps(L)`，除以 `s` 折回 q 的量纲。调用方把它写进报告，
    好让"这个 B 到底准到第几位"是个记录在案的数，而不是默认它精确。
    """

    import numpy as np

    return float(np.spacing(_RHO_AFFINE_BIAS) / float(slope_scale))


# =============================================================================
# 顶层：一条轨迹 → 闭式解 → payload + weights + manifest
# =============================================================================
def _resolve_vocabulary(atomic_numbers, teacher_z_table, log) -> tuple[int, ...]:
    """EXP-033 §5 P1 item 4：词表 = teacher z-table ∩ 体系，不是 `unique(体系)`。

    §2.3A 实测那两个词表**从来没对齐过**：体系推出来混进一个 MACE-OFF24 根本表示
    不了的 Na；反过来 teacher 覆盖的 F/P/Br/I 又不在词表里。没给 teacher 时退回
    体系元素（保持旧行为），但在报告里写明走的是哪条。
    """

    system_elements = tuple(sorted(set(int(z) for z in atomic_numbers)))
    if teacher_z_table is None:
        log("  [refit] 未给 teacher z-table ⟹ 词表退回体系元素（§5 P1 item 4 未生效）")
        return system_elements
    teacher = set(int(z) for z in teacher_z_table)
    missing = [z for z in system_elements if z not in teacher]
    if missing:
        raise RefitError(
            f"体系里有 teacher 表示不了的元素 {missing}（teacher z-table 有 {len(teacher)} 种）。"
            "这正是「模型不覆盖这个体系」的证据，必须停下来报告，不是跳过那些帧 —— "
            "跳过等于把本该触发停止的信号变成拟合时看不见的样本。"
        )
    return system_elements


def _probe_reduced_energies(
    *, system, topology, positions, box_vectors, ligand_indices,
    trajectory, states, temperature_kelvin: float, platform_name: str, log,
):
    """逐态软核 probe 现算约化势能 —— **不读** `energies.npy`。

    EXP-033 §5 P1 item 2 要求对账 `energies.npy` 的 LRC 与单位。这里干脆不读它：
    自己按 `(lam_coul, lam_vdw)` 逐态求 `U^sc_k`，单位与 LRC 口径都由本函数一手定义，
    于是那类"同类不对齐骗过 5.5σ"的风险在这条路上不存在
    （`docs/STAGE2_SOLVENT_LEG_ERROR_BUDGET.md`）。

    也不走 `TraditionalMBARAnalyzer.compute_u_kn`：它是传统 REMD 腿的重加权入口，
    会给目标能量补 Beutler 软核缺的 1/V LRC 尾项，因此带一道"轨迹必须固定盒"的门，
    而预平衡是 NPT、盒在动。那个尾项是 V 和 λ 的光滑函数，跟模型看得见的局部几何
    无关，模型既学不到也不该学它。
    """

    import numpy as np
    import openmm
    from openmm import unit

    from abfe_preoptimizer import ACESoftcorePotential, build_aces_probe_system_dual_lambda

    n_ligand = len(list(ligand_indices))
    softcore = ACESoftcorePotential.from_dict(
        ACESoftcorePotential.optimize_alpha(n_ligand)
    )
    probe_system = build_aces_probe_system_dual_lambda(
        system, list(ligand_indices), softcore,
        fixed_lam_coul=0.0, fixed_lam_vdw=1.0,
        topology=topology, positions=positions, box_vectors=box_vectors,
    )
    for index in reversed(range(probe_system.getNumForces())):
        if "Barostat" in type(probe_system.getForce(index)).__name__:
            probe_system.removeForce(index)
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    context = openmm.Context(
        probe_system, integrator, openmm.Platform.getPlatformByName(str(platform_name))
    )
    try:
        kt = 0.008314462618 * float(temperature_kelvin)
        frames = int(trajectory.n_frames)
        block = np.zeros((frames, len(states)), dtype=np.float64)
        for frame in range(frames):
            context.setPeriodicBoxVectors(
                *(trajectory.unitcell_vectors[frame] * unit.nanometer)
            )
            context.setPositions(trajectory.xyz[frame] * unit.nanometer)
            for index, (lam_c, lam_v) in enumerate(states):
                context.setParameter("lam_coul", float(lam_c))
                context.setParameter("lam_vdw", float(lam_v))
                energy = context.getState(getEnergy=True).getPotentialEnergy()
                block[frame, index] = energy.value_in_unit(unit.kilojoule_per_mole) / kt
        if not np.all(np.isfinite(block)):
            raise RefitError("probe 算出来的约化势能里有非有限值")
        log(f"  [refit] probe 完成：{frames} 帧 × {len(states)} 态")
        return block
    finally:
        del context, integrator


def closed_form_refit(
    *,
    system,
    topology,
    positions,
    box_vectors,
    ligand_indices: Sequence[int],
    trajectory_path: str | Path,
    lambdas_vdw: Sequence[float],
    lambdas_coul: Sequence[float] | None = None,
    sampling_lambda_coul: float = 1.0,
    sampling_lambda_vdw: float = 1.0,
    temperature_kelvin: float = 300.0,
    platform_name: str = "CUDA",
    output_dir: str | Path,
    ligand_name: str = "MOL",
    topology_cif: str | Path,
    ligand_indices_path: str | Path,
    system_xml: str | Path | None = None,
    teacher_z_table: Sequence[int] | None = None,
    n_folds: int = 3,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """EXP-033 §5 P1 的一次闭式重训。无 GPU 依赖（platform 可给 CPU），不 import torch。

    帧源是调用方给的那条轨迹，**有多少帧用多少**、不抽稀、不设帧数下限。
    """

    import numpy as np

    _log = _log_to(log)
    out = Path(output_dir)
    (out / "resources").mkdir(parents=True, exist_ok=True)

    from .openmm_plugin import topology_atomic_numbers
    from .softlift import derived_r1_config

    atomic_numbers = topology_atomic_numbers(topology, system=system)
    vocabulary = _resolve_vocabulary(atomic_numbers, teacher_z_table, _log)
    ligand_ids = [int(v) for v in ligand_indices]
    n_ligand = len(ligand_ids)
    capacities = _derived_capacities(n_ligand)

    import mdtraj

    trajectory = mdtraj.load(str(trajectory_path), top=str(topology_cif))
    n_frames = int(trajectory.n_frames)
    if n_frames < n_folds:
        raise RefitError(f"{Path(trajectory_path).name} 只有 {n_frames} 帧，切不出 {n_folds} 折")
    _log(f"  [refit] 帧源 {Path(trajectory_path).name}：{n_frames} 帧全用")

    lam_vdw = [float(v) for v in lambdas_vdw]
    lam_coul = (
        [float(v) for v in lambdas_coul] if lambdas_coul is not None
        else [0.0] * len(lam_vdw)
    )
    states = list(zip(lam_coul, lam_vdw)) + [
        (float(sampling_lambda_coul), float(sampling_lambda_vdw))
    ]
    reduced = _probe_reduced_energies(
        system=system, topology=topology, positions=positions,
        box_vectors=box_vectors, ligand_indices=ligand_ids, trajectory=trajectory,
        states=states, temperature_kelvin=temperature_kelvin,
        platform_name=platform_name, log=_log,
    )
    target_u = reduced[:, : len(lam_vdw)]
    sampling_u = reduced[:, len(lam_vdw)]
    adjacent_gap_reduced = np.diff(target_u, axis=1)
    log_importance = sampling_u[:, None] - target_u
    delta_a, a_k = a_k_schedule(lam_vdw)

    n_types = len(vocabulary)
    # 协议身份：决定"这份权重是按什么口径拟出来的"的全部输入。
    # 不含帧内容 —— 帧是数据不是协议；换帧要重拟，但那由调用方的缓存判据管。
    protocol_sha256 = hashlib.sha256(
        json.dumps(
            {
                "chain": "local_residual.refit",
                "version": 1,
                "lambdas_vdw": lam_vdw,
                "lambdas_coul": lam_coul,
                "sampling_state": [float(sampling_lambda_coul), float(sampling_lambda_vdw)],
                "type_vocabulary": list(vocabulary),
                "n_ligand_atoms": n_ligand,
                "capacities": capacities,
                "n_folds": int(n_folds),
                "A_definition": "sin_squared_pi_lambda_vdw",
                "regularizer": "second_difference_smoothness_prior_in_r",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    config = derived_r1_config(
        protocol_sha256, type_vocabulary=vocabulary, n_ligand_atoms=n_ligand, **capacities
    )
    centers = np.linspace(0.0, config.outer_cutoff_angstrom, config.n_radial_basis)
    width = config.outer_cutoff_angstrom / max(config.n_radial_basis - 1, 1)
    z_to_type = {int(z): index for index, z in enumerate(vocabulary)}
    all_types = np.asarray([z_to_type[int(z)] for z in atomic_numbers], dtype=np.int64)
    ligand_sorted = np.asarray(sorted(ligand_ids), dtype=np.int64)
    environment = np.asarray(
        [i for i in range(len(atomic_numbers)) if i not in set(ligand_ids)], dtype=np.int64
    )

    features = np.zeros((n_frames, n_types * n_types * config.n_radial_basis))
    realized = {"edges": 0, "environment_atoms": 0, "neighbors": 0}
    for frame in range(n_frames):
        row, frame_realized = frame_basis_features(
            positions_angstrom=trajectory.xyz[frame] * 10.0,
            box_angstrom=trajectory.unitcell_vectors[frame] * 10.0,
            ligand_indices=ligand_sorted,
            environment_indices=environment,
            ligand_type_index=all_types[ligand_sorted],
            environment_type_index=all_types,
            n_types=n_types,
            radial_centers=centers,
            radial_width=width,
            inner_cutoff_angstrom=config.inner_cutoff_angstrom,
            outer_cutoff_angstrom=config.outer_cutoff_angstrom,
        )
        features[frame] = row
        for key in realized:
            realized[key] = max(realized[key], frame_realized[key])
    _log(f"  [refit] 特征完成：{features.shape[1]} 维；实测用量 {realized}")

    edges = [round(index * n_frames / n_folds) for index in range(n_folds + 1)]
    fold_index = np.zeros(n_frames, dtype=np.int64)
    for fold in range(n_folds):
        fold_index[edges[fold]: edges[fold + 1]] = fold

    theta, fit_report = solve_closed_form(
        features=features,
        adjacent_gap_reduced=adjacent_gap_reduced,
        log_importance_unnormalized=log_importance,
        delta_a=delta_a,
        fold_index=fold_index,
        n_types=n_types,
        n_radial=config.n_radial_basis,
        log=_log,
    )

    raw = features @ theta
    # §5 P1 item 5 第一条：减掉常数。逐态常数被 f_k 吸收，对 ΔG 无影响。
    offset = float(np.mean(raw))
    centered = raw - offset
    bound = float(config.b_max_reduced)
    saturated = float(np.mean(np.abs(centered) > bound))
    _log(
        f"  [refit] 线性 B 量程 [{centered.min():.2f}, {centered.max():.2f}] kT；"
        f"b_max={bound:.1f}；tanh 咬合帧比例 {saturated:.1%}"
    )

    result = {
        "theta": theta,
        "features": features,
        "raw_offset_reduced": offset,
        "linear_basis_reduced": centered,
        "tanh_saturated_fraction": saturated,
        "config": config,
        "vocabulary": vocabulary,
        "capacities": capacities,
        "realized_capacity_usage": realized,
        "fit_report": fit_report,
        "a_k": a_k,
        "delta_a": delta_a,
        "n_frames": n_frames,
        "trajectory_path": str(trajectory_path),
        "protocol_sha256": protocol_sha256,
    }
    return result


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_refit_payload(
    result: Mapping[str, Any],
    *,
    output_dir: str | Path,
    ligand_topology_indices: Sequence[int],
    protocol_sha256: str,
    min_deployed_spread_fraction: float = 0.05,
    log: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """把闭式解写成 loader 能吃的 payload + weights。

    产物与 `abfe_scripts/export_exp025_g1_reference_payload.py` 同一套 schema
    （`openmm_plugin.load_r1_payload` 逐字段读它），区别只在张量是解出来的、
    不是从 `.pt` 里搬的。`source_checkpoint.sha256` 记的是**拟合输入的身份**而不是
    某个 checkpoint —— 这条链上没有 checkpoint，写个假的更糟。
    """

    import numpy as np

    _log = _log_to(log)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = result["config"]
    theta = np.asarray(result["theta"], dtype=np.float64)
    n_types = len(config.type_vocabulary)
    n_radial = int(config.n_radial_basis)

    # rho 的斜率折进 pair_weight：s 按实测 |q| 上界定，保证 SiLU 段仍然精确。
    pair_weight = theta.reshape(n_types, n_types, n_radial)
    basis = np.asarray(result["linear_basis_reduced"], dtype=np.float64)
    q_bound = float(np.abs(basis).max())
    # 留 1% 余量：直接取 `MAX_Q / q_bound` 会让 `s·q_bound` 落在 15.000000000000002，
    # 下面那道 `>= 45` 就会因为一个 ULP 判否。
    slope_scale = (
        1.0 if q_bound <= 0.0 else min(1.0, 0.99 * _RHO_AFFINE_MAX_Q / q_bound)
    )
    if not affine_rho_is_exact(basis, slope_scale):
        raise RefitError(f"仿射 rho 构造失效：|q|max={q_bound:.3g}, s={slope_scale:.3g}")
    error_floor = affine_rho_absolute_error_floor(slope_scale)
    bound = float(config.b_max_reduced)
    if error_floor > 1.0e-6 * bound:
        raise RefitError(
            f"仿射 rho 的精度下限 {error_floor:.3g} kT 相对 b_max={bound:.1f} 已不可忽略"
            f"（|q|max={q_bound:.3g} 逼得斜率缩到 {slope_scale:.3g}）"
        )

    # §5 P1 item 5：拟合的是**线性** B，部署的是 `b_max·tanh(B/b_max)`。
    # §2.3B 实测线性量程能到 ±28~40 kT 而 b_max 只有 10 ⟹ tanh 改形是**预期**会发生的。
    #
    # 这里**不**拿"部署的 B 要逼近拟合的 B"当硬门。`B_φ` 的职责是让配体解耦时整体
    # 别散架（提高 λ 态之间与相邻窗口共享态那条缝上的混合），不是一个要逐点复现的
    # 物理模型 —— 压到 tanh 的饱和段仍然是朝正确方向的一记大推力。按精度标准硬拦，
    # 拦掉的是本来能跑的 run。改形量照量、照记、大了就喊，但不停。
    deployed = bound * np.tanh(basis / bound)
    spread = float(np.sqrt(np.mean(basis**2)))
    deviation = float(np.sqrt(np.mean((deployed - basis) ** 2)))
    fidelity = deviation / spread if spread > 0 else 0.0
    if fidelity > 0.10:
        _log(
            f"  [refit] ⚠️ tanh 把 B 改形 {fidelity:.1%}：线性量程 "
            f"[{basis.min():.1f}, {basis.max():.1f}] kT vs b_max={bound:.1f}。"
            "这本身不算错（饱和段仍是朝正确方向的推力），但若增强效果不及预期，"
            "先看这个数 —— 正解是按配体尺寸重定 b_max_reduced，不是放宽判据。"
        )

    # 🔑 真正该 fail-closed 的是这条：tanh 把**所有**帧压到同一侧 ⟹ 部署出去的 B
    # 近似常数。逐态常数会被 f_k 整个吸收掉，于是这个增强项静悄悄归零 —— 跑完一整轮
    # GPU 才发现"开了等于没开"。用部署后的离散度判，因为那才是运行时真正看到的量。
    # 判别量是**塌缩比**，不是绝对离散度：`B` 本来就小（信号弱）和 `tanh 把它压平了`
    # 是两回事。前者是个结果、交给 U3 上机判；后者是这份 payload 根本不该导出。
    deployed_spread = float(np.std(deployed))
    linear_spread = float(np.std(basis))
    collapse = deployed_spread / linear_spread if linear_spread > 0 else 0.0
    if collapse < min_deployed_spread_fraction:
        raise RefitError(
            f"tanh 把 B 的离散度压掉到只剩 {collapse:.1%}"
            f"（{linear_spread:.3g} → {deployed_spread:.3g} kT，b_max={bound:.1f}，"
            f"下限 {min_deployed_spread_fraction:.0%}）⟹ 帧被全压到同一侧，"
            "B 已退化成一个会被 f_k 完全吸收的常数，开了等于没开。"
            "拒绝导出这样一份 payload —— 这是「这批帧撑不起拟合」的证据，不是精度问题。"
        )
    if deployed_spread < 0.05 * bound:
        _log(
            f"  [refit] ⚠️ 部署后的 B 离散度只有 {deployed_spread:.3g} kT"
            f"（b_max={bound:.1f}）：拟合没在这批帧上找到多少信号。"
            "这不是错误、也不拦 —— 但增强效果大概率很弱，由 U3 上机判。"
        )

    state: dict[str, Any] = {
        "pair_weight": pair_weight,
        "radial_centers": np.linspace(0.0, config.outer_cutoff_angstrom, n_radial),
        "radial_width": np.asarray(
            config.outer_cutoff_angstrom / max(n_radial - 1, 1), dtype=np.float64
        ),
    }
    state.update(_affine_rho_state(n_types, slope_scale))

    blob = bytearray()
    manifest: list[dict[str, Any]] = []
    for name in sorted(state):
        array = np.ascontiguousarray(state[name], dtype="<f8")
        raw = array.tobytes()
        manifest.append({
            "name": name,
            "shape": list(array.shape),
            "dtype": "float64_little_endian",
            "byte_offset": len(blob),
            "byte_count": len(raw),
            "sha256": _sha256_bytes(raw),
        })
        blob.extend(raw)
    blob = bytes(blob)

    weights_path = out / "r1_model_weights_f64.bin"
    payload_path = out / "r1_model_payload_v1.json"
    weights_path.write_bytes(blob)
    body = {
        "schema_version": "exp025-g1-reference-payload-v1",
        "experiment_id": "EXP-033",
        "source_checkpoint": {
            "path": "(closed-form refit; 这条链上没有 checkpoint)",
            "sha256": protocol_sha256,
            "held_out_run": None,
            "seed": None,
        },
        "config": {
            "schema_version": config.schema_version,
            "rung": config.rung,
            "type_vocabulary": list(config.type_vocabulary),
            "n_ligand_atoms": config.n_ligand_atoms,
            "n_radial_basis": n_radial,
            "n_channels": config.n_channels,
            "pair_dim": config.pair_dim,
            "context_dim": config.context_dim,
            "inner_cutoff_angstrom": config.inner_cutoff_angstrom,
            "outer_cutoff_angstrom": config.outer_cutoff_angstrom,
            "b_max_reduced": config.b_max_reduced,
            "max_environment_atoms": config.max_environment_atoms,
            "max_edges": config.max_edges,
            "max_neighbors_per_ligand": config.max_neighbors_per_ligand,
            "no_contact_output": config.no_contact_output,
            "protocol_sha256": protocol_sha256,
        },
        "closed_form_refit": {
            "chain": "local_residual.refit",
            "reference": "docs/EXP-033_LOCAL_RESIDUAL_REFIT_2026-09-10.md §5 P1",
            "trajectory_path": result["trajectory_path"],
            "n_frames": result["n_frames"],
            "ridge_selected": result["fit_report"]["ridge_selected"],
            "relative_improvement": result["fit_report"]["relative_improvement"],
            "raw_offset_reduced": result["raw_offset_reduced"],
            "tanh_saturated_fraction": result["tanh_saturated_fraction"],
            "rho_affine_slope_scale": slope_scale,
            "rho_affine_absolute_error_floor": error_floor,
            "tanh_deployment_distortion": fidelity,
            "deployed_basis_std_reduced": deployed_spread,
            "tanh_spread_collapse_ratio": collapse,
            "realized_capacity_usage": result["realized_capacity_usage"],
            "acceptance_note": (
                "离线 gap-variance 改善只是「值得上机的信号」，不是验收。"
                "真验收是 EXP-027 U3 口径：window-0 utility + ΔG 一致性，"
                "且两臂各自独立标定并冻结自己的 f_k（EXP-033 §4）。"
            ),
        },
        "ligand_topology_indices": [int(v) for v in ligand_topology_indices],
        "weights_file": {
            "name": weights_path.name,
            "sha256": _sha256_bytes(blob),
            "byte_count": len(blob),
            "layout": (
                "flat concatenation of tensors in the order listed in tensor_manifest, "
                "little-endian float64"
            ),
        },
        "tensor_manifest": manifest,
    }
    payload_path.write_text(json.dumps(body, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    # 回读一道：loader 自己读得动，才算产出成功。
    from .openmm_plugin import load_r1_payload

    reloaded = load_r1_payload(payload_path, weights_path)
    if not np.array_equal(reloaded.pair_weight, pair_weight):
        raise RefitError("payload 回读后的 pair_weight 与解出来的不一致")
    _log(f"  [refit] payload 已写出并回读通过：{payload_path}")
    return {"payload": payload_path, "weights": weights_path}

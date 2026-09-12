"""`_lj_tail_correction_sigma_resolved_moments` 的逐位等价 oracle。

## 为什么是逐位而不是 allclose

这个函数产出的 `(sigma_nm, s6_per_sigma, s12_per_sigma)` 是 LRC 尾修正的几何量，
一路进 `ibs_wrapper.lj_tail_lrc_coeff_kj_mol`，再进每帧的约化能量。它**不该**因为
一次实现优化而变哪怕一个 ulp —— 变了就意味着旧采样缓存与新代码算出来的
u_kn 不再是同一个哈密顿量，而这种偏差小到不会触发任何门。

所以这里冻结改写之前的 dense 实现当 oracle，用 `.tobytes()` 比对。
`allclose` 会放过正是我们要排除的那类漂移。

## 冻结的是什么

`_dense_oracle_v3` 是 2026-09-09 改写**之前**的 `ibs_engine.
_lj_tail_correction_sigma_resolved_moments` 的数学部分逐字拷贝（去掉 unit 解包）。
改写的动机是内存：dense 版本会同时物化 n_ligand × n_environment 的
`sigma_ij` / `eps_ij` / `sigma_key` / `inverse` / 两个权重数组。

改写必须保住下面这些顺序，任何一条动了都不再逐位等价：

* ligand 外层、environment 内层（决定累加次序）；
* 同样的 `round(..., 9)` 分组键；
* 权重用**未 round** 的 sigma；
* `np.unique` 的升序输出（`searchsorted` 命中的必须是同一组 bin 且同序）；
* 按原始 pair 次序累加（`np.bincount` 是按输入下标顺序累加的，
  逐行 `np.add.at` 只有在行序与行内序都保持时才等价）。
"""
from __future__ import annotations

import numpy as np
import pytest

from openmm import unit

import ibs_engine


def _dense_oracle_v3(sigma_lig, eps_lig, sigma_env, eps_env):
    """改写前 dense 实现的数学部分，逐字冻结。不要"顺手优化"这个函数。"""
    sigma_ij = (0.5 * (sigma_lig[:, None] + sigma_env[None, :])).ravel()
    eps_ij = np.sqrt(eps_lig[:, None] * eps_env[None, :]).ravel()
    sigma_key = np.round(sigma_ij, 9)
    sigma_nm, inverse = np.unique(sigma_key, return_inverse=True)
    n_bins = sigma_nm.shape[0]
    s6 = np.bincount(inverse, weights=eps_ij * sigma_ij ** 6, minlength=n_bins)
    s12 = np.bincount(inverse, weights=eps_ij * sigma_ij ** 12, minlength=n_bins)
    return sigma_nm.astype(np.float64), s6.astype(np.float64), s12.astype(np.float64)


class _P:
    """`getParticleParameters()` 返回值的最小替身：(charge, sigma, epsilon)。"""

    __slots__ = ("_items",)

    def __init__(self, sigma_nm: float, eps_kj: float):
        self._items = (
            0.0 * unit.elementary_charge,
            float(sigma_nm) * unit.nanometer,
            float(eps_kj) * unit.kilojoule_per_mole,
        )

    def __getitem__(self, index):
        return self._items[index]


def _build(sigma_lig, eps_lig, sigma_env, eps_env):
    """拼出 `all_params` + 两组索引，索引刻意不连续、不排序。"""
    n_l, n_e = len(sigma_lig), len(sigma_env)
    # 配体放在中间、环境分布在两侧，确保实现不能依赖"配体是前 N 个"。
    all_params = []
    env_head = n_e // 2
    for k in range(env_head):
        all_params.append(_P(sigma_env[k], eps_env[k]))
    for k in range(n_l):
        all_params.append(_P(sigma_lig[k], eps_lig[k]))
    for k in range(env_head, n_e):
        all_params.append(_P(sigma_env[k], eps_env[k]))
    ligand_indices = list(range(env_head, env_head + n_l))
    # 尾段只有 n_e - env_head 个环境原子，不是 n_e。
    environment_indices = list(range(env_head)) + list(
        range(env_head + n_l, env_head + n_l + (n_e - env_head))
    )
    assert len(all_params) == n_l + n_e
    assert len(environment_indices) == n_e
    return all_params, ligand_indices, environment_indices


def _real_forcefield_values(rng, n, kind):
    """取真实力场量级的离散 sigma/epsilon —— 分组行为依赖"取值很少但会重复"。"""
    if kind == "ligand":
        sigma_pool = np.array(
            [0.107, 0.2471, 0.3399, 0.3399, 0.3250, 0.2960, 0.3564, 0.1960],
            dtype=np.float64,
        )
        eps_pool = np.array(
            [0.0657, 0.0657, 0.4577, 0.3598, 0.7113, 0.8786, 1.0460, 0.0870],
            dtype=np.float64,
        )
    else:
        sigma_pool = np.array(
            [0.3151, 0.0000, 0.3324, 0.4401, 0.2960, 0.3399], dtype=np.float64
        )
        eps_pool = np.array(
            [0.6364, 0.0000, 0.0116, 0.4184, 0.8786, 0.4577], dtype=np.float64
        )
    idx = rng.integers(0, sigma_pool.size, size=n)
    return sigma_pool[idx].copy(), eps_pool[idx].copy()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "n_lig,n_env",
    [
        (41, 20000),      # 本仓真实量级（配体 41 原子 / 大水盒），缩了环境规模保测试时长
        (100, 5000),      # 大配体
        (1, 1),           # 退化：单对
        (1, 7),           # 退化：单配体原子
        (7, 1),           # 退化：单环境原子
        (3, 4),           # 小样例，人工可核对
    ],
)
def test_sigma_resolved_moments_are_bit_identical_to_dense_oracle(n_lig, n_env):
    rng = np.random.default_rng(20260909)
    sigma_lig, eps_lig = _real_forcefield_values(rng, n_lig, "ligand")
    sigma_env, eps_env = _real_forcefield_values(rng, n_env, "env")

    expected = _dense_oracle_v3(sigma_lig, eps_lig, sigma_env, eps_env)

    all_params, ligand_indices, environment_indices = _build(
        sigma_lig, eps_lig, sigma_env, eps_env
    )
    actual = ibs_engine._lj_tail_correction_sigma_resolved_moments(
        all_params, ligand_indices, environment_indices
    )

    names = ("sigma_nm", "s6_per_sigma", "s12_per_sigma")
    for name, got, want in zip(names, actual, expected):
        assert got.dtype == want.dtype == np.float64, name
        assert got.shape == want.shape, (
            f"{name} 形状变了：{got.shape} != {want.shape}（分组集合或其顺序被改动）"
        )
        assert got.tobytes() == want.tobytes(), (
            f"{name} 与冻结的 dense oracle **不逐位相同**。\n"
            f"  最大绝对差 = {np.max(np.abs(got - want)):.3e}\n"
            "  这不是可以用 allclose 放过的差异：这三个量进 LRC 系数、再进每帧约化能量，"
            "变一个 ulp 就意味着旧采样缓存与新代码不是同一个哈密顿量，"
            "而这种偏差小到不会触发任何门。\n"
            "  改写必须保住：ligand 外层/environment 内层的累加次序、round(...,9) 分组键、"
            "权重用未 round 的 sigma、np.unique 的升序 bin、按原始 pair 次序累加。"
        )


@pytest.mark.cpu_only
def test_zero_epsilon_environment_still_groups_by_sigma():
    """ε=0 的环境原子（本仓真实存在：虚拟位点/dummy）不得改变分组集合。

    它们贡献 0 权重，但仍然参与 sigma 分组 —— 若实现"顺手"把 ε=0 的 pair 跳过，
    `sigma_nm` 的 bin 集合就会变短，与 oracle 不再同形。
    """
    sigma_lig = np.array([0.3399, 0.2471], dtype=np.float64)
    eps_lig = np.array([0.4577, 0.0657], dtype=np.float64)
    sigma_env = np.array([0.3151, 0.0000, 0.4401], dtype=np.float64)
    eps_env = np.array([0.6364, 0.0000, 0.4184], dtype=np.float64)

    expected = _dense_oracle_v3(sigma_lig, eps_lig, sigma_env, eps_env)
    all_params, lig, env = _build(sigma_lig, eps_lig, sigma_env, eps_env)
    actual = ibs_engine._lj_tail_correction_sigma_resolved_moments(all_params, lig, env)

    for got, want in zip(actual, expected):
        assert got.tobytes() == want.tobytes()

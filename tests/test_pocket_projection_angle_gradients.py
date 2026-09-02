"""[0831issue #4] `OrbBoreschEstimator._compute_geom_gradients` 的角度解析梯度回归。

旧实现把两个端点的分子写反了（且各自保留了原来的 1/|ba| vs 1/|bc| 缩放，两处错误
不互相抵消），有限差分复核相对误差 130%~200%。该梯度经
`run_pocket_force_projection` → `_apply_hybrid_filter` 直接决定 Boresch 的
kthetaA/kthetaB 力常数，`--boresch-source auto` / `orb_simple` 两条路都会走到。

这里不构造 `OrbBoreschEstimator` 实例（它的 `__init__` 需要 torch + openmmml +
MACE-OFF 许可），直接把未绑定方法作用在一个只提供本方法所需属性的最小 stub 上。
"""

import numpy as np
import pytest

pytest.importorskip("openmm")

import abfe_core  # noqa: E402


ANGLE_SLOTS = ((1, 0, 3), (0, 3, 4))


class _Stub:
    """`_compute_geom_gradients` 只用到 self（不读任何属性）。"""


def _compute(r_anchors):
    return abfe_core.OrbBoreschEstimator._compute_geom_gradients(
        _Stub(), np.asarray(r_anchors, dtype=float)
    )


def _theta(a, b, c):
    ba, bc = a - b, c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.arccos(np.clip(cos_val, -1.0, 1.0)))


def _distance(a, b):
    return float(np.linalg.norm(b - a))


def _fd_gradient(r_anchors, value_fn, slots, eps=1e-6):
    """对指定 slot 做中心差分，返回 (6, 3) 梯度（未涉及的 slot 为 0）。"""
    grad = np.zeros((6, 3))
    for slot in slots:
        for d in range(3):
            plus = r_anchors.copy()
            minus = r_anchors.copy()
            plus[slot, d] += eps
            minus[slot, d] -= eps
            grad[slot, d] = (value_fn(plus) - value_fn(minus)) / (2.0 * eps)
    return grad


@pytest.mark.parametrize("seed", [20260901, 20260902, 20260903, 20260904, 20260905])
def test_angle_gradients_match_finite_difference(seed):
    rng = np.random.default_rng(seed)
    # 随机但分得开的 6 个锚点，避免共线（sinA→0 时函数走的是另一条 if 分支）。
    r_anchors = rng.normal(scale=0.5, size=(6, 3))

    q, grads = _compute(r_anchors)

    for i, (sa, sb, sc) in enumerate(ANGLE_SLOTS):
        g_idx = i + 1
        assert np.isclose(
            q[g_idx], _theta(r_anchors[sa], r_anchors[sb], r_anchors[sc]), atol=1e-12
        )
        sin_theta = np.sin(q[g_idx])
        if sin_theta <= 1e-3:
            pytest.skip("近共线构型走的是零梯度分支，不在本用例范围")

        fd = _fd_gradient(
            r_anchors,
            lambda pos, sa=sa, sb=sb, sc=sc: _theta(pos[sa], pos[sb], pos[sc]),
            (sa, sb, sc),
        )
        scale = max(np.max(np.abs(fd)), 1e-12)
        # FD 自身精度 ~1e-8；旧实现在这里的相对误差是 130%~200%。
        assert np.max(np.abs(grads[g_idx] - fd)) / scale < 1e-5, (
            f"角度 {g_idx} 的解析梯度与有限差分不符：\n"
            f"analytic=\n{grads[g_idx]}\nfd=\n{fd}"
        )

        # 未参与该角的 slot 必须严格为 0（别把梯度写到别人的槽位）。
        for other in set(range(6)) - {sa, sb, sc}:
            assert np.allclose(grads[g_idx, other], 0.0)


def test_angle_gradients_sum_to_zero_translational_invariance():
    """整体平移不改变角度 → 三个 slot 的梯度之和必须为 0。"""
    rng = np.random.default_rng(20260906)
    r_anchors = rng.normal(scale=0.5, size=(6, 3))
    _, grads = _compute(r_anchors)
    for i in range(len(ANGLE_SLOTS)):
        assert np.allclose(grads[i + 1].sum(axis=0), 0.0, atol=1e-10)


def test_distance_gradient_still_matches_finite_difference():
    """slot 0 的距离梯度本来就是对的，一并钉住，防止改角度时误伤。"""
    rng = np.random.default_rng(20260907)
    r_anchors = rng.normal(scale=0.5, size=(6, 3))
    q, grads = _compute(r_anchors)
    assert np.isclose(q[0], _distance(r_anchors[0], r_anchors[3]))
    fd = _fd_gradient(
        r_anchors, lambda pos: _distance(pos[0], pos[3]), (0, 3)
    )
    assert np.max(np.abs(grads[0] - fd)) < 1e-6


def test_wrong_historical_formula_is_actually_wrong():
    """把旧式重算一遍，确认它确实偏离 FD——否则本回归无意义。"""
    rng = np.random.default_rng(20260908)
    r_anchors = rng.normal(scale=0.5, size=(6, 3))
    sa, sb, sc = ANGLE_SLOTS[0]
    a, b, c = r_anchors[sa], r_anchors[sb], r_anchors[sc]
    ba, bc = a - b, c - b
    nba, nbc = np.linalg.norm(ba) + 1e-10, np.linalg.norm(bc) + 1e-10
    cos_a = np.clip(np.dot(ba, bc) / (nba * nbc), -1, 1)
    sin_a = np.sqrt(1 - cos_a**2) + 1e-10
    assert sin_a > 1e-3
    legacy_dbda = (cos_a * bc / nbc - ba / nba) / (nba * sin_a)

    fd = _fd_gradient(
        r_anchors,
        lambda pos: _theta(pos[sa], pos[sb], pos[sc]),
        (sa, sb, sc),
    )
    scale = max(np.max(np.abs(fd)), 1e-12)
    assert np.max(np.abs(legacy_dbda - fd[sa])) / scale > 0.1

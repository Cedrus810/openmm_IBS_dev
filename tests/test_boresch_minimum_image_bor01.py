"""BOR-01：Boresch 六锚点几何的 minimum-image 口径必须只有一份。

事故形状：写进限制势的 `calc_boresch_from_last_frame` 用裸 `norm(a-b)`，
而校验用的 `_check_boresch_geometry_safe` 自己解了缠 ⟹ 锚点对跨周期边界时
提交的 r0 静默差一个盒矢量（~4–5 nm），**校验函数反而看不出分歧**。

三条判据（来自 docs/TODO.md §7）：
  1. 两份实现在同一份跨边界坐标上给出逐位相同的 r0/θA/θB；
  2. 跨边界坐标上，不解缠的旧口径差约一个盒矢量、解缠后不差；
  3. 不跨边界的坐标上 r0/θ/φ **逐位不变**（否则作废全部 Boresch 缓存）。
"""
import numpy as np
import pytest

from abfe_core import calc_boresch_from_last_frame, unwrap_boresch_anchors_nm

BOX = np.diag([5.0, 5.0, 5.0])
REC = [0, 1, 2]
LIG = [3, 4, 5]

# 连续（不跨边界）的一组锚点，θ 远离奇点、r0 在 [0.3, 2.0] nm 硬门内。
POS_CONTIGUOUS = np.array([
    [2.30, 2.50, 2.50],   # H0
    [1.90, 2.85, 2.40],   # H1
    [1.50, 2.60, 2.05],   # H2
    [3.20, 2.60, 2.70],   # L0
    [3.55, 2.30, 3.00],   # L1
    [3.90, 2.55, 3.35],   # L2
], dtype=float)


def _wrapped_across_x(pos, shift=2.4):
    """整体平移后按盒回卷 —— 物理上同一个构象，但 L 那半跑到盒的另一侧。"""
    return np.mod(pos + np.array([shift, 0.0, 0.0]), np.diag(BOX))


def _geom_tuple(eq):
    return tuple(eq[k] for k in ("r0", "thetaA0", "thetaB0", "phiA0", "phiB0", "phiC0"))


def test_contiguous_geometry_is_bitwise_unchanged():
    """判据 3：没跨边界时带不带 box 必须逐位相同，否则既有平衡值全部作废。"""
    without = _geom_tuple(calc_boresch_from_last_frame(POS_CONTIGUOUS, REC, LIG))
    with_box = _geom_tuple(
        calc_boresch_from_last_frame(POS_CONTIGUOUS, REC, LIG, box_vectors=BOX)
    )
    assert with_box == without, "解缠在不跨边界时不是恒等映射"


def test_unwrapped_geometry_matches_the_physical_conformer():
    """判据 2：回卷撕开的构型，解缠后必须还原成同一个几何量。"""
    ref = _geom_tuple(calc_boresch_from_last_frame(POS_CONTIGUOUS, REC, LIG))
    wrapped = _wrapped_across_x(POS_CONTIGUOUS)

    fixed = _geom_tuple(
        calc_boresch_from_last_frame(wrapped, REC, LIG, box_vectors=BOX)
    )
    assert np.allclose(fixed, ref, atol=1e-9), "解缠后几何与原构象不一致"

    # 旧口径（不传 box）必须真的错，否则这条测试没有鉴别力。
    with pytest.raises(RuntimeError, match="r0"):
        # r0 差约一个盒矢量 ⟹ 直接撞上 [3, 20] Å 的硬门
        calc_boresch_from_last_frame(wrapped, REC, LIG)


def test_both_implementations_share_one_unwrap():
    """判据 1：校验那份（ibs_engine）与提交那份用的是同一份解缠实现。"""
    import ibs_engine

    assert ibs_engine.unwrap_boresch_anchors_nm is unwrap_boresch_anchors_nm

    wrapped = _wrapped_across_x(POS_CONTIGUOUS)
    H0, H1, H2, L0, L1, L2 = unwrap_boresch_anchors_nm(
        wrapped[REC], wrapped[LIG], BOX
    )
    eq = calc_boresch_from_last_frame(wrapped, REC, LIG, box_vectors=BOX)

    # 逐位相同，不是 allclose。
    assert float(np.linalg.norm(H0 - L0)) == eq["r0"]

    def ang(a, b, c):
        ba, bc = a - b, c - b
        return float(np.arccos(np.clip(
            np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10),
            -1.0, 1.0,
        )))

    # 两份实现的 angle() 差一个 +1e-10 的分母护栏，只能到机器精度。
    assert ang(H1, H0, L0) == pytest.approx(eq["thetaA0"], abs=1e-12)
    assert ang(H0, L0, L1) == pytest.approx(eq["thetaB0"], abs=1e-12)
    assert H2 is not None and L2 is not None


def test_unwrap_without_box_is_identity():
    """box=None = 调用方声明坐标已连续，原样返回。"""
    out = unwrap_boresch_anchors_nm(POS_CONTIGUOUS[REC], POS_CONTIGUOUS[LIG], None)
    assert np.array_equal(out, POS_CONTIGUOUS)


def test_unwrap_rejects_wrong_anchor_count():
    with pytest.raises(ValueError, match="3\\+3"):
        unwrap_boresch_anchors_nm(POS_CONTIGUOUS[:2], POS_CONTIGUOUS[3:], BOX)

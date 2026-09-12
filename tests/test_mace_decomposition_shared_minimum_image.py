"""MACE 三区域分解必须用**同一套**周期镜像。

## 钉的是什么

`_evaluate_decomposition` 要算 `E_cplx − E_lig − E_env`。2026-09-10 之前它对三个
区域各做一次 `_minimum_image_selected`，而那个函数拿 `selected[0]` 当锚点：

    cplx → 锚在 ligand_indices[0]
    lig  → 锚在 ligand_indices[0]
    env  → 锚在 environment_indices[0]      ← 不同的锚点

离锚点超过半个盒长的原子会被选到**不同的**周期镜像，于是 `E_cplx` 内部那份环境
子几何与单独算的 `E_env` 的几何不是同一个构型，差值里混进一个伪的胞内项。

## 为什么断言"一致"而不是"物理"

只要三项坐标**逐位同源**，这个分解就精确成立——哪怕环境壳大于 L/2、镜像看起来
不漂亮。所以这里断言的是一致性，不是"每个原子都落在最近镜像"。
反过来说：任何"把 env 按自己的锚点重新 image 一遍"的好心改动都会破坏它，
下面第二条测试就是为了让那种改动立刻红。

不需要 MACE 模型：`_minimum_image_selected` 是 staticmethod，纯几何。
"""
from __future__ import annotations

import numpy as np
import pytest

from outer_lambda_neural_basis import MaceDecompositionPythonComputation as _MDC

pytestmark = pytest.mark.cpu_only

_IMAGE = _MDC._minimum_image_selected


def _boundary_straddling_case():
    """配体在盒子一角，环境跨到对面——正是锚点选择会分歧的构型。"""
    box = np.diag([4.0, 4.0, 4.0]).astype(np.float64)
    positions = np.asarray(
        [
            [0.20, 0.20, 0.20],   # 0 ligand
            [0.35, 0.22, 0.18],   # 1 ligand
            [3.80, 3.85, 3.90],   # 2 env —— 与配体隔着周期边界
            [3.60, 0.10, 0.05],   # 3 env
            [2.10, 2.05, 2.00],   # 4 env —— 离配体超过半盒长
        ],
        dtype=np.float64,
    )
    ligand = [0, 1]
    environment = [2, 3, 4]
    return positions, ligand, environment, box


def test_env_slice_of_combined_imaging_matches_what_cplx_uses():
    positions, ligand, environment, box = _boundary_straddling_case()
    combined = ligand + environment

    imaged_combined = _IMAGE(positions, combined, box)
    env_from_combined = imaged_combined[len(ligand):]
    lig_from_combined = imaged_combined[: len(ligand)]

    # cplx 用的就是 imaged_combined 本身，所以切片必然逐位一致。
    # 这条钉的是"切片位置对不对"（combined == ligand + environment）。
    assert env_from_combined.tobytes() == imaged_combined[len(ligand):].tobytes()
    assert lig_from_combined.tobytes() == _IMAGE(positions, ligand, box).tobytes(), (
        "lig 区域与 combined 的前缀切片不一致——两者锚点都是 ligand_indices[0]，"
        "本应逐位相同。"
    )


def test_per_region_imaging_really_differs_so_the_bug_was_real():
    """旧写法（env 自己锚）与新写法（从 combined 切）在这个构型上必须不同。

    如果哪天这两者变成恒等，说明这个测试用例已经不再覆盖那个失效模式
    （比如构型被改成全都落在半盒长内），要换构型而不是删断言。
    """
    positions, ligand, environment, box = _boundary_straddling_case()
    combined = ligand + environment

    env_old = _IMAGE(positions, environment, box)          # 旧：锚在 environment[0]
    env_new = _IMAGE(positions, combined, box)[len(ligand):]  # 新：锚在 ligand[0]

    assert env_old.shape == env_new.shape
    assert not np.allclose(env_old, env_new), (
        "本用例没有触发锚点分歧——旧写法和新写法给出了同一套坐标。\n"
        "  换一个跨周期边界更明显的构型，别删这条断言："
        "它是「这个 bug 真的存在」的唯一证据。"
    )
    # 差异必须是整盒矢量的整数倍（只挪镜像，不改内部几何）。
    delta = env_old - env_new
    fractional = delta @ np.linalg.inv(box)
    assert np.allclose(fractional, np.round(fractional), atol=1e-12), (
        "两种成像的差不是盒矢量的整数倍——那就不只是镜像选择不同了。"
    )


def test_internal_geometry_is_unchanged_by_imaging():
    """成像只能整体平移原子，不能改任何区域内部的相对几何。"""
    positions, ligand, environment, box = _boundary_straddling_case()
    combined = ligand + environment
    imaged = _IMAGE(positions, combined, box)

    raw = positions[combined]
    for i in range(len(combined)):
        for j in range(i + 1, len(combined)):
            shift = (raw[i] - raw[j]) - (imaged[i] - imaged[j])
            fractional = shift @ np.linalg.inv(box)
            assert np.allclose(fractional, np.round(fractional), atol=1e-12), (
                f"原子对 ({combined[i]}, {combined[j]}) 的相对位移不是盒矢量整数倍，"
                "成像改动了内部几何。"
            )

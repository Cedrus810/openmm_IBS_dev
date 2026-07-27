"""ATT-09 回归：热力学循环闭合只能有一份实现，且不得漏项。

改之前同一个公式在四处独立维护，而且**它们并不等价**：

  | 位置 | Boresch | APBS |
  |---|---|---|
  | `runabfe.main()`                      | 已内含   | ✅ 加了 |
  | `runabfe.run_traditional_mode()`      | 显式减   | ❌ 没有 |
  | `runabfe.run_post_analysis()`         | 条件置零 | ✅ 加了 |
  | `ABFEPipeline.run_full_abfe_loop()`   | 已内含   | ❌ 没有 |

后两条路径对带电配体会静默漏掉整项有限尺寸静电修正——这是数值错误，不是整洁性问题。
统一之后那两条路径的输出**会变**，那是修复。

APBS 本身由离线的 `apbs_correction.py`（prepare → run → collect）算出，
collect 的输出里直接给出要传的 `--apbs-correction-kj-mol`；流程内只消费这个标量。
"""

import ast
import math
from pathlib import Path

import pytest

from abfe_core import combine_binding_free_energy

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 解析 toy cycle
# ---------------------------------------------------------------------------


def test_toy_cycle_has_an_exact_analytic_answer():
    """给定四个数，ΔG_bind 只有一个正确值；钉死到 1e-12。

    ΔG_complex = 100（含释放项），ΔG_solvent = 60，ΔG_APBS = -3
      → ΔG_bind = 60 - 100 + (-3) = -43
    """
    out = combine_binding_free_energy(
        dg_complex_kJ_mol=100.0,
        dg_solvent_kJ_mol=60.0,
        err_complex_kJ_mol=3.0,
        err_solvent_kJ_mol=4.0,
        dg_boresch_kJ_mol=12.0,
        boresch_already_included_in_complex=True,
        apbs_correction_kJ_mol=-3.0,
    )
    assert out["delta_G_bind_uncorrected_kJ_mol"] == pytest.approx(-40.0, abs=1e-12)
    assert out["delta_G_bind_kJ_mol"] == pytest.approx(-43.0, abs=1e-12)
    # 3-4-5 直角三角形：误差合并必须是求积和方根，不是相加。
    assert out["total_error_kJ_mol"] == pytest.approx(5.0, abs=1e-12)
    assert out["delta_G_bind_kcal_mol"] == pytest.approx(-43.0 / 4.184, abs=1e-12)


def test_boresch_is_subtracted_exactly_once_when_not_baked_in():
    """complex 腿不含释放项时减一次；含了就不减。两者的差必须正好是 ΔG_release。"""
    common = dict(
        dg_complex_kJ_mol=100.0,
        dg_solvent_kJ_mol=60.0,
        dg_boresch_kJ_mol=12.0,
        apbs_correction_kJ_mol=0.0,
    )
    baked = combine_binding_free_energy(
        boresch_already_included_in_complex=True, **common
    )
    not_baked = combine_binding_free_energy(
        boresch_already_included_in_complex=False, **common
    )
    assert baked["delta_G_bind_kJ_mol"] == pytest.approx(-40.0, abs=1e-12)
    assert not_baked["delta_G_bind_kJ_mol"] == pytest.approx(-52.0, abs=1e-12)
    assert baked["delta_G_bind_kJ_mol"] - not_baked["delta_G_bind_kJ_mol"] == (
        pytest.approx(12.0, abs=1e-12)
    )
    # 记账字段必须区分"物理量"与"公式里真减掉的那一项"。
    assert baked["boresch_correction_kJ_mol"] == 12.0
    assert baked["boresch_term_subtracted_kJ_mol"] == 0.0
    assert not_baked["boresch_term_subtracted_kJ_mol"] == 12.0


def test_two_conventions_agree_when_describing_the_same_physical_system():
    """同一个体系用两种约定表述，ΔG_bind 必须相同。

    约定 A：complex 腿报 88（不含释放项），释放项 12 单列。
    约定 B：complex 腿报 100（= 88 + 12，已内含）。
    这是"减且只减一次"的独立交叉验证，不是上一条的重述。
    """
    a = combine_binding_free_energy(
        dg_complex_kJ_mol=88.0, dg_solvent_kJ_mol=60.0,
        dg_boresch_kJ_mol=12.0, boresch_already_included_in_complex=False,
    )
    b = combine_binding_free_energy(
        dg_complex_kJ_mol=100.0, dg_solvent_kJ_mol=60.0,
        dg_boresch_kJ_mol=12.0, boresch_already_included_in_complex=True,
    )
    assert a["delta_G_bind_kJ_mol"] == pytest.approx(
        b["delta_G_bind_kJ_mol"], abs=1e-12
    )


def test_apbs_enters_additively_and_defaults_to_zero():
    """APBS 是纯加性外部项；不传时必须恰好等于 0，不能悄悄变成别的。"""
    without = combine_binding_free_energy(
        dg_complex_kJ_mol=100.0, dg_solvent_kJ_mol=60.0,
    )
    with_apbs = combine_binding_free_energy(
        dg_complex_kJ_mol=100.0, dg_solvent_kJ_mol=60.0,
        apbs_correction_kJ_mol=-7.5,
    )
    assert without["apbs_correction_kJ_mol"] == 0.0
    assert with_apbs["delta_G_bind_kJ_mol"] - without["delta_G_bind_kJ_mol"] == (
        pytest.approx(-7.5, abs=1e-12)
    )
    # 漏掉 APBS 正是 site 2/site 4 的原缺陷；这条钉住"漏了就看得见"。
    assert without["delta_G_bind_kJ_mol"] != with_apbs["delta_G_bind_kJ_mol"]


def test_sign_convention_a_real_binder_gives_negative_delta_g():
    """真结合物：口袋里去耦更贵（ΔG_complex > ΔG_solvent）→ ΔG_bind 为负。

    符号弄反过一次（旧代码写成 ΔG_complex - ΔG_solvent），这条专门钉方向。
    """
    out = combine_binding_free_energy(
        dg_complex_kJ_mol=140.0, dg_solvent_kJ_mol=95.0,
    )
    assert out["delta_G_bind_kJ_mol"] < 0.0
    assert out["delta_G_bind_kJ_mol"] == pytest.approx(-45.0, abs=1e-12)


def test_none_errors_are_treated_as_zero_not_as_nan():
    out = combine_binding_free_energy(
        dg_complex_kJ_mol=10.0, dg_solvent_kJ_mol=5.0,
        err_complex_kJ_mol=None, err_solvent_kJ_mol=2.0,
        apbs_correction_kJ_mol=None,
    )
    assert math.isfinite(out["total_error_kJ_mol"])
    assert out["total_error_kJ_mol"] == pytest.approx(2.0, abs=1e-12)
    assert out["apbs_correction_kJ_mol"] == 0.0


# ---------------------------------------------------------------------------
# 四个调用点确实都改用了这唯一实现
# ---------------------------------------------------------------------------

CALL_SITES = [
    ("runabfe.py", "main"),
    ("runabfe.py", "run_traditional_mode"),
    ("runabfe.py", "run_post_analysis"),
    ("abfe_pipeline.py", "run_full_abfe_loop"),
]


def _function_node(filename, func_name):
    tree = ast.parse((REPO_ROOT / filename).read_text(encoding="utf-8"), filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return node
    raise AssertionError(f"{filename} 里找不到 {func_name}")


@pytest.mark.parametrize("filename, func_name", CALL_SITES)
def test_call_site_uses_the_single_implementation(filename, func_name):
    node = _function_node(filename, func_name)
    called = {
        n.func.id
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "combine_binding_free_energy" in called, (
        f"{filename}::{func_name} 又在手写循环闭合公式；四处必须共用同一个实现，"
        "否则 APBS/Boresch 的处理会再次分叉（这正是 ATT-09 的原缺陷）"
    )


@pytest.mark.parametrize("filename, func_name", CALL_SITES)
def test_call_site_passes_an_apbs_term(filename, func_name):
    """每个调用点都必须显式传 apbs_correction_kJ_mol——漏传就是回到原缺陷。"""
    node = _function_node(filename, func_name)
    for call in ast.walk(node):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
            continue
        if call.func.id != "combine_binding_free_energy":
            continue
        kwargs = {kw.arg for kw in call.keywords if kw.arg}
        assert "apbs_correction_kJ_mol" in kwargs, (
            f"{filename}::{func_name} 调 combine_binding_free_energy 时没传 "
            "apbs_correction_kJ_mol；带电配体会静默漏掉有限尺寸静电修正"
        )
        assert "boresch_already_included_in_complex" in kwargs, (
            f"{filename}::{func_name} 没显式声明 Boresch 是否已内含；"
            "这个开关决定要不要再减一次，绝不能靠默认值"
        )

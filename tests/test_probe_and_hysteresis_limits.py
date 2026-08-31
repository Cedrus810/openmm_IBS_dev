"""三个已知代码缺陷的回归。

1) `abfe_core.STAGE2_HYSTERESIS_MAX_SIGMA` 曾是**死常量**：只被写进 provenance，
   没有任何代码执行它。修复后它是湿/干双起点门的唯一真源，且本文件禁止
   ibs_engine 另立同义数值——两份定义会让"provenance 记录的阈值"与"实际生效的
   阈值"分叉，那正是它最初变成死常量的同一类问题。
2) `IBSSampler._build_probe_context` 用 `gid = 16 + idx` 给每个 λ 态分配 force
   group，OpenMM 上限 31 ⟹ 单次最多 16 个态。第 17 个态原先直接抛
   `OpenMMException: Force group must be between 0 and 31`，既不说明是哪一层的
   限制，也不告诉调用方怎么办。修复后是构建前的可读错误。
3) `build_ibs_dual_system` 的 K<=16 限制来自 **Group-1 IBS 混合偏置力**的
   CustomCVForce 32-CV 上限，**不是** λ 态数的物理上限。逐态独立固定-λ 采样
   不用那个偏置力，不应被它约束——错误信息必须讲清楚这一点。
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# 1) 迟滞阈值：单一真源
# --------------------------------------------------------------------------

def test_hysteresis_threshold_has_a_single_source_of_truth():
    import abfe_core
    import ibs_engine

    assert ibs_engine.ENDPOINT_WET_DRY_MAX_SIGMA is abfe_core.STAGE2_HYSTERESIS_MAX_SIGMA


def test_ibs_engine_does_not_redefine_the_threshold_numerically():
    """禁止写成 `ENDPOINT_WET_DRY_MAX_SIGMA = 2.0` 这类独立字面量。"""
    src = (REPO / "ibs_engine.py").read_text(encoding="utf-8")
    m = re.search(r"^ENDPOINT_WET_DRY_MAX_SIGMA\s*=\s*(.+)$", src, re.M)
    assert m, "找不到 ENDPOINT_WET_DRY_MAX_SIGMA 的定义"
    assert m.group(1).strip() == "STAGE2_HYSTERESIS_MAX_SIGMA", (
        f"必须直接别名 abfe_core 的常量，实际是 {m.group(1)!r}"
    )


def test_the_gate_actually_executes_the_threshold():
    """它必须真的被用来判定，而不是只出现在 provenance 里。"""
    from ibs_engine import endpoint_wet_dry_hysteresis_gate

    ok = {"converged": True, "delta_G_kJ_mol": 0.0, "delta_G_sigma_kJ_mol": 0.30}
    far = {"converged": True, "delta_G_kJ_mol": 5.0, "delta_G_sigma_kJ_mol": 0.30}
    cav = {
        "a": {"init_mode": "wet", "wet_fraction": 0.6, "n_wet_dry_transitions": 10},
        "b": {"init_mode": "dry", "wet_fraction": 0.4, "n_wet_dry_transitions": 9},
    }
    assert endpoint_wet_dry_hysteresis_gate(ok, ok, cav)["passed"] is True
    bad = endpoint_wet_dry_hysteresis_gate(ok, far, cav)
    assert bad["passed"] is False
    assert "wet_dry_delta_exceeds_sigma_gate" in bad["failed_checks"]
    import abfe_core
    assert bad["max_sigma"] == pytest.approx(abfe_core.STAGE2_HYSTERESIS_MAX_SIGMA)


# --------------------------------------------------------------------------
# 2) 探针 Context 的 force-group 预算
# --------------------------------------------------------------------------

def test_probe_force_group_budget_is_derived_not_hardcoded():
    import ibs_engine as ie

    assert ie.OPENMM_MAX_FORCE_GROUP == 31
    assert ie.PROBE_FORCE_GROUP_BASE == 16
    assert ie.PROBE_MAX_LAMBDA_STATES == 16
    src = (REPO / "ibs_engine.py").read_text(encoding="utf-8")
    assert "gid = PROBE_FORCE_GROUP_BASE + idx" in src, "分配处必须用常量，不能写死 16"
    assert "gid = 16 + idx" not in src


def test_probe_context_rejects_more_than_sixteen_states_readably():
    """第 17 个态必须是**构建前**的可读错误，而不是 OpenMM 的裸异常。"""
    pytest.importorskip("openmm")
    import openmm
    from openmm import unit

    import ibs_engine as ie

    class _Wrapper:
        prefix = "test"
        _int_cv_force_xmls = [
            openmm.XmlSerializer.serialize(openmm.CustomExternalForce("0.0*x"))
        ] * (ie.PROBE_MAX_LAMBDA_STATES + 1)

    sysobj = openmm.System()
    sysobj.addParticle(12.0)
    L = 3.0
    sysobj.setDefaultPeriodicBoxVectors(
        openmm.Vec3(L, 0, 0), openmm.Vec3(0, L, 0), openmm.Vec3(0, 0, L)
    )
    ctx = openmm.Context(
        sysobj, openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    with pytest.raises(RuntimeError, match="探针 Context 无法容纳"):
        ie.IBSSampler(ctx, 17, 298.15 * unit.kelvin,
                      prefix="test", ibs_wrapper=_Wrapper())


# --------------------------------------------------------------------------
# 3) 32-CV 上限的归属必须讲清楚
# --------------------------------------------------------------------------

def test_cv_limit_error_says_it_is_the_ibs_bias_not_a_physical_limit():
    src = (REPO / "ibs_engine.py").read_text(encoding="utf-8")
    i = src.index("单个 IBS ensemble 的 lambda 状态过多")
    msg = src[i:i + 1400]
    assert "Group-1 IBS 混合偏置力" in msg, "必须说明限制来自混合偏置力"
    assert "不是** λ 态数本身的物理上限" in msg
    assert "INDEPENDENT_ENDPOINT_PROTOCOL_VERSION" in msg, "必须指出独立采样不受此限"
    assert "分块调用" in msg, "必须给出可行的绕法"


def test_the_two_limits_are_not_confused_with_each_other():
    """探针的 16 态上限与 IBS 偏置的 16 态上限是两条独立限制，数值巧合相同。"""
    import ibs_engine as ie

    assert ie.PROBE_MAX_LAMBDA_STATES == ie.IBS_DUAL_MAX_LAMBDA_STATES == 16
    src = (REPO / "ibs_engine.py").read_text(encoding="utf-8")
    i = src.index("n_cv > PROBE_MAX_LAMBDA_STATES")
    assert "与 build_ibs_dual_system 里那条" in src[i - 900:i], (
        "探针检查处必须注明它与 IBS 偏置那条限制不是同一回事"
    )

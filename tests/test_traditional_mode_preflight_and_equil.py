"""P1-13 / P1-14：traditional 模式的基线预平衡与协议声明接入。

## P1-13 缺陷是什么（已修，本文件现在是契约测试）

`run_traditional_mode` 的基线预平衡只发生在 `if config.boresch:` 里（借道
`resolve_boresch_restraint` 的"无条件先跑一次"）。显式 `--no-boresch` 时，
临时 Boresch ABFEPipeline 根本不会创建，两条腿直接 setPositions/setVelocities
进 REMD——原始或仅居中的坐标直接进入生产采样。

## P1-14 缺陷是什么（已修）

`main()` 在 normal preflight 之前就直接分流 traditional；膜 system_type、
`dispersion_protocol`、`forcefield_family` 这些声明只可能进入临时 Boresch
ABFEPipeline，`--no-boresch` 时连临时检查也没有——用户声明的协议被静默
忽略、按默认路线跑。

## 修法（2026-08-30）

- P1-13：complex/solvent 两条腿在建 TraditionalABFEPipeline 之前，各自经统一
  fingerprint / `equilibrium_is_done`（含 P1-07 严格校验）/ `pre_equilibrate`
  完成（或严格复用）基线预平衡，独立于 Boresch 开关；
- P1-14：`_assert_traditional_protocol_supported(config)` 在建任何 Context 之前
  对非默认协议声明 fail closed。

## 不要这样让本文件变绿

把预平衡挪回 `if config.boresch:` 里、或让 `_assert_traditional_protocol_supported`
对膜/非默认色散声明打 warning 后继续跑。
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

import runabfe
from runabfe import _assert_traditional_protocol_supported

ROOT = Path(__file__).resolve().parents[1]
RUNABFE_PATH = ROOT / "runabfe.py"


class _FakeConfig:
    def __init__(self, **values):
        self._values = {
            "system_type": None,
            "membrane": None,
            "dispersion_protocol": None,
            "forcefield_family": None,
        }
        self._values.update(values)

    def get(self, key, default=None):
        return self._values.get(key, default)


# ---------------------------------------------------------------------------
# 1. P1-14：协议声明 fail closed
# ---------------------------------------------------------------------------


def test_default_declarations_are_allowed():
    _assert_traditional_protocol_supported(_FakeConfig())
    _assert_traditional_protocol_supported(
        _FakeConfig(system_type="soluble", forcefield_family="auto")
    )


@pytest.mark.parametrize(
    "declaration",
    [
        {"system_type": "membrane"},
        {"membrane": {"apl_target_nm2": 0.62}},
        {"dispersion_protocol": "lj_pme"},
        {"dispersion_protocol": "membrane_inhomogeneous"},
        {"forcefield_family": "charmm36"},
    ],
)
def test_unsupported_declarations_fail_closed(declaration):
    with pytest.raises(RuntimeError, match="traditional 模式不支持"):
        _assert_traditional_protocol_supported(_FakeConfig(**declaration))


def test_protocol_check_is_called_before_any_context_creation():
    """`_assert_traditional_protocol_supported` 必须先于预平衡/REMD 的任何建
    Context 调用（源码顺序守护）。"""
    src = RUNABFE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    run_traditional_mode = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_traditional_mode"
    )
    call_lines = {}
    for node in ast.walk(run_traditional_mode):
        if isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            call_lines.setdefault(name, []).append(node.lineno)

    assert "_assert_traditional_protocol_supported" in call_lines
    check_line = min(call_lines["_assert_traditional_protocol_supported"])
    assert "pre_equilibrate" in call_lines, "traditional 必须执行基线预平衡（P1-13）"
    for heavy in ("pre_equilibrate", "ABFEPipeline", "TraditionalABFEPipeline"):
        assert check_line < min(call_lines[heavy]), (
            f"协议声明检查必须先于 {heavy}（P1-14：建 Context 前明确拒绝）"
        )


# ---------------------------------------------------------------------------
# 2. P1-13：基线预平衡独立于 Boresch，两条腿都要
# ---------------------------------------------------------------------------


def test_baseline_pre_equilibration_is_not_nested_under_boresch_gate():
    """pre_equilibrate 调用不得位于 `if config.boresch` 守卫之内。"""
    src = RUNABFE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    run_traditional_mode = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_traditional_mode"
    )
    parents = {}
    for parent in ast.walk(run_traditional_mode):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    def _guarded_by_boresch(node):
        ancestor = parents.get(node)
        while ancestor is not None:
            if isinstance(ancestor, ast.If):
                test_src = ast.unparse(ancestor.test)
                if "config.boresch" in test_src:
                    return True
            ancestor = parents.get(ancestor)
        return False

    equil_calls = [
        node for node in ast.walk(run_traditional_mode)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "pre_equilibrate"
    ]
    assert len(equil_calls) >= 2, (
        "complex/solvent 两条腿都必须有基线预平衡（P1-13）"
    )
    for node in equil_calls:
        assert not _guarded_by_boresch(node), (
            "基线预平衡被嵌在 `if config.boresch` 里 —— --no-boresch 时两条腿"
            "会再次跳过预平衡（P1-13 回归）"
        )


def test_both_legs_equilibrate_before_their_remd_pipelines_are_built():
    """两条腿的预平衡都必须先于对应 TraditionalABFEPipeline 的构建。"""
    src = RUNABFE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    run_traditional_mode = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_traditional_mode"
    )
    first_equil = min(
        node.lineno for node in ast.walk(run_traditional_mode)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "pre_equilibrate"
    )
    pipeline_builds = [
        node.lineno for node in ast.walk(run_traditional_mode)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "TraditionalABFEPipeline"
    ]
    assert len(pipeline_builds) == 2
    assert first_equil < min(pipeline_builds), (
        "基线预平衡必须先于任何 traditional REMD pipeline 构建（P1-13："
        "不得让原始坐标进入生产采样）"
    )


def test_baseline_gate_uses_strict_equilibrium_is_done():
    """两条腿的完成判断都带 expected_fingerprint + 目标 Simulation 校验。"""
    src = RUNABFE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    run_traditional_mode = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_traditional_mode"
    )
    equilibrium_calls = [
        node for node in ast.walk(run_traditional_mode)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "equilibrium_is_done"
    ]
    assert len(equilibrium_calls) >= 2, "两条腿各需要一个完成判断"
    for node in equilibrium_calls:
        keywords = {kw.arg for kw in node.keywords}
        assert "expected_fingerprint" in keywords, "完成判断必须带协议指纹"
        assert "simulation" in keywords, (
            "完成判断必须传目标 Simulation（P1-07 严格 checkpoint 校验）"
        )

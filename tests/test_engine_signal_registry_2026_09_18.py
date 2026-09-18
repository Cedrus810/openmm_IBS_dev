"""引擎控制面异常 → 控制器语义：注册表 + 完整性校验的回归钉。

洞：`abfe_pipeline` 的两处 handler 各写了一个**手写二元元组**
（`IBSValidationBudgetIndeterminateError` + `IBSWarmupConvergenceError`），
而引擎有 5 个控制面异常。漏掉的 `ExistingEnsembleRequiresRescueAudit` 的消息
自己写着 `route it to rescue/provenance audit`，七个抛出点全部 fail-closed，
其中四条守卫是 `non_mutating_v1` —— **自治模式的常态策略**。
它落进 `except Exception` ⟹ `_finish("FAILED") + raise` ⟹ 一个已知、可分类的
控制器结局被伪装成执行器崩溃。

⚠️ 只补一个 dict 不够：没登记的新异常照旧落进 `except Exception`。所以配套立了
`IBSControlPlaneError` 基类，import 期拿 `__subclasses__()` 跟注册表对账。
"""
import gc
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import abfe_pipeline as pipe  # noqa: E402
import ibs_engine as ie  # noqa: E402
from abfe_preoptimizer import EXIT_SPECS  # noqa: E402


def test_every_control_plane_exception_is_registered():
    """引擎声明的每一个控制面异常都必须在注册表里有处置。"""
    declared = {c.__name__ for c in ie.IBSControlPlaneError.__subclasses__()}
    assert declared == set(pipe.ENGINE_SIGNAL_SPECS), (
        "注册表与 `IBSControlPlaneError.__subclasses__()` 不一致 ⟹ "
        "未登记的异常会静默落进 `except Exception`"
    )


def test_registry_completeness_check_actually_fires():
    """加一个没登记的控制面异常 ⟹ 校验必须抛，而不是静默放过。

    这是这套机制的**全部价值**：注册表本身挡不住"忘了登记"，靠的是这道对账。
    """
    # ⚠️ 造完必须真的回收掉：`__subclasses__()` 返回的是**新列表**，对它 `.clear()`
    # 是空操作，脏子类会留在类型层级里，把上面那条用例按执行顺序打红。
    # CPython 的子类表是弱引用 ⟹ 删名字 + 强制 gc 才真的摘掉。
    try:
        _Unregistered = type(
            "_UnregisteredControlPlaneError", (ie.IBSControlPlaneError,), {})
        with pytest.raises(RuntimeError, match="未登记"):
            pipe._assert_engine_signal_registry_complete()
    finally:
        del _Unregistered
        gc.collect()

    # 回收干净了 ⟹ 校验重新通过。这一条同时保证本用例不污染别的用例。
    pipe._assert_engine_signal_registry_complete()
    assert "_UnregisteredControlPlaneError" not in {
        c.__name__ for c in ie.IBSControlPlaneError.__subclasses__()
    }


def test_control_plane_exceptions_are_still_runtime_errors():
    """插入基类是纯增量：既有 `except RuntimeError` 的行为一个 bit 不变。"""
    for cls in ie.IBSControlPlaneError.__subclasses__():
        assert issubclass(cls, RuntimeError)


def test_halt_entries_name_a_real_terminal_exit():
    """`HALT` 档给的出口必须真的存在于 `EXIT_SPECS`，而且必须是终态。"""
    halts = {k: v for k, v in pipe.ENGINE_SIGNAL_SPECS.items()
             if v.handler == "HALT"}
    assert halts, "至少 ExistingEnsembleRequiresRescueAudit 是 HALT 档"
    for name, spec in halts.items():
        assert spec.exit_ in EXIT_SPECS, f"{name} 的出口不在 EXIT_SPECS 里"
        assert EXIT_SPECS[spec.exit_].terminal, (
            f"{name} 是 HALT 档却配了非终态出口 {spec.exit_}"
        )


def test_existing_ensemble_audit_is_halt_on_an_invalid_path():
    """它的处置必须是终态 `HALT_INVALID_INPUT` / `PATH_INVALID`，不是 REDECIDE。

    走 `REDECIDE` 的后果不是死循环，是**归因错**：盘面一字节不变 ⟹ no-op 台账
    把动作挡掉 ⟹ 最后以 `NO_FEASIBLE_ACTION` 收场，把「ensemble 身份不成立」
    说成「动作推不动」—— 而那正是人工接手时最需要分清的一件事。
    """
    spec = pipe.ENGINE_SIGNAL_SPECS["ExistingEnsembleRequiresRescueAudit"]
    assert spec.handler == "HALT"
    assert spec.exit_ == "HALT_INVALID_INPUT"
    # scope 不用发出点传：`EXIT_SPECS` 给的默认就是 PATH_INVALID（绕过去无意义）
    assert EXIT_SPECS[spec.exit_].default_scope == "PATH_INVALID"


def test_not_routed_is_declared_explicitly_not_by_omission():
    """「刻意不路由」必须登记成 NOT_ROUTED —— 缺席读不出是决定还是遗忘。"""
    spec = pipe.ENGINE_SIGNAL_SPECS["IBSIncompleteStageCoverageError"]
    assert spec.handler == "NOT_ROUTED"
    assert spec.exit_ is None and spec.signal is None

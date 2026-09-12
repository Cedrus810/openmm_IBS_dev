"""`step_guard` 两条出口的行为断言 —— 无需 GPU。

## 为什么需要这条测试

2026-09-03 精简测试步进 schedule 时留了一个回退：炸了就回滚这一小段、步长减半重做。
但它只在 `sim.step()` **返回之后**才检查有限性，而 OpenMM 的 CUDA 平台是在 `step()`
内部的原子重排里检查坐标并抛 `OpenMMException("Particle coordinate is NaN")` 的 ——
真实失败模式下那个检查永远走不到，回退一次都没触发过。2026-09-09 的 OpenFF
burn-in benchmark 里 10 个 run 全部这样在窗口 0 的偏置预热里炸穿。

这条测试直接盯 `_step_with_chunk_rollback` 的两种失败路径，不需要真 Context。
"""
import numpy as np
import openmm
import pytest
from openmm import unit

from step_guard import guarded_step, step_with_chunk_rollback


class _FakeState:
    def __init__(self, finite: bool):
        self._finite = finite

    def getPotentialEnergy(self):
        return (1.0 if self._finite else float("nan")) * unit.kilojoule_per_mole

    def getForces(self, asNumpy=False):
        v = np.zeros((2, 3)) if self._finite else np.full((2, 3), np.inf)
        return v * unit.kilojoule_per_mole / unit.nanometer

    def getPositions(self, asNumpy=False):
        return np.zeros((2, 3)) * unit.nanometer


class _FakeIntegrator:
    def __init__(self, dt_ps):
        self.dt = dt_ps * unit.picoseconds

    def getStepSize(self):
        return self.dt

    def setStepSize(self, dt):
        self.dt = dt


class _FakeContext:
    def __init__(self, owner):
        self.owner = owner
        self.restores = 0

    def getState(self, **kw):
        return _FakeState(self.owner.finite)

    def setState(self, state):
        self.restores += 1


class _FakeSim:
    """前 `n_bad` 次 step() 失败，之后成功。`mode` 决定失败长什么样。"""

    def __init__(self, n_bad, mode, dt_ps=0.002):
        self.integrator = _FakeIntegrator(dt_ps)
        self.context = _FakeContext(self)
        self.n_bad = n_bad
        self.mode = mode
        self.calls = 0
        self.finite = True

    def step(self, n):
        self.calls += 1
        bad = self.calls <= self.n_bad
        self.finite = not bad
        if bad and self.mode == "raise":
            raise openmm.OpenMMException("Particle coordinate is NaN")


@pytest.mark.parametrize("mode", ["raise", "silent"])
def test_rollback_rescues_and_restores_step_size(mode):
    sim = _FakeSim(n_bad=2, mode=mode, dt_ps=0.002)
    rescued = step_with_chunk_rollback(sim, 400, "t", chunk=200)

    assert rescued == 2, "两次失败应各触发一次回退"
    assert sim.context.restores == 2, "每次失败都必须回滚到该段开始前的构型/速度"
    # 段过了就恢复入口步长，减半只在失败的那一段内生效
    assert sim.integrator.getStepSize().value_in_unit(unit.picoseconds) == pytest.approx(0.002)


@pytest.mark.parametrize("mode", ["raise", "silent"])
def test_rollback_gives_up_at_min_dt_with_diagnostics(mode):
    sim = _FakeSim(n_bad=10**6, mode=mode, dt_ps=0.002)
    seen = []
    with pytest.raises(RuntimeError, match="仍出现非有限"):
        step_with_chunk_rollback(
            sim, 200, "t", chunk=200, min_dt_ps=1.0e-5, on_exhausted=seen.append
        )
    assert seen == ["t"], "耗尽步长下限时必须先打诊断再抛"
    assert sim.integrator.getStepSize().value_in_unit(unit.picoseconds) == pytest.approx(0.002)


def test_guarded_step_converts_openmm_exception_and_keeps_cause():
    """生产/采样路径：不救、不改步长，但必须变成带上下文的 RuntimeError。"""
    sim = _FakeSim(n_bad=1, mode="raise", dt_ps=0.002)
    seen = []
    with pytest.raises(RuntimeError, match="窗口 3 冻结验证") as ei:
        guarded_step(sim, 500, "窗口 3 冻结验证", on_error=seen.append)
    assert isinstance(ei.value.__cause__, openmm.OpenMMException), "原始异常不能被吞掉"
    assert seen == ["窗口 3 冻结验证"], "抛之前必须先打诊断"
    assert sim.integrator.getStepSize().value_in_unit(unit.picoseconds) == pytest.approx(0.002)


def test_guarded_step_is_transparent_when_nothing_blows_up():
    sim = _FakeSim(n_bad=0, mode="raise")
    guarded_step(sim, 500, "正常")
    assert sim.calls == 1


# ---------------------------------------------------------------------------
# 下面这些 guard 尚未接入生产（见 step_guard 模块 docstring 的"接入状态"表），
# 但实现已经在了，行为先钉住，免得接线那天才发现语义不对。
# ---------------------------------------------------------------------------

from step_guard import (  # noqa: E402
    finite_state_check,
    guarded_context,
    guarded_deserialize,
    guarded_get_parameter,
    guarded_minimize,
    guarded_platform,
    guarded_set_parameter,
    require_parameters,
)



pytestmark = pytest.mark.cpu_only

class _ParamContext:
    """只认识 known 里那些全局参数，其余一律像 OpenMM 那样抛。"""

    def __init__(self, known):
        self.known = dict(known)

    def getParameter(self, name):
        if name not in self.known:
            raise openmm.OpenMMException(f"Called getParameter() with invalid parameter name: {name}")
        return self.known[name]

    def setParameter(self, name, value):
        if name not in self.known:
            raise openmm.OpenMMException(f"Called setParameter() with invalid parameter name: {name}")
        self.known[name] = value


def test_finite_state_check_catches_nan_in_forces_only():
    """能量还是有限的、力已经 NaN —— 只看能量的检查会放过这种。"""
    class _Ctx:
        def getState(self, **kw):
            class S:
                def getPotentialEnergy(self_):
                    return 1.0 * unit.kilojoule_per_mole
                def getPositions(self_, asNumpy=False):
                    return np.zeros((2, 3)) * unit.nanometer
                def getForces(self_, asNumpy=False):
                    return np.full((2, 3), np.nan) * unit.kilojoule_per_mole / unit.nanometer
            return S()
    ok, _ = finite_state_check(_Ctx())
    assert ok is False
    assert finite_state_check(_Ctx(), check_forces=False)[0] is True


def test_finite_state_check_never_raises_when_context_is_unreadable():
    class _Broken:
        def getState(self, **kw):
            raise openmm.OpenMMException("Particle coordinate is NaN")
    ok, why = finite_state_check(_Broken())
    assert ok is False and "读取 State 失败" in why


def test_guarded_minimize_catches_both_divergence_and_silent_nan():
    class _Sim:
        def __init__(self, raise_it, finite_after):
            self.raise_it, self.finite_after = raise_it, finite_after
            self.context = self
        def minimizeEnergy(self, **kw):
            if self.raise_it:
                raise openmm.OpenMMException("Particle coordinate is NaN")
        def getState(self, **kw):
            class S:
                def getPotentialEnergy(s_):
                    return (1.0 if self.finite_after else float("inf")) * unit.kilojoule_per_mole
                def getPositions(s_, asNumpy=False):
                    return np.zeros((2, 3)) * unit.nanometer
                def getForces(s_, asNumpy=False):
                    return np.zeros((2, 3)) * unit.kilojoule_per_mole / unit.nanometer
            return S()

    with pytest.raises(RuntimeError, match="发散"):
        guarded_minimize(_Sim(True, True), "窗口 0 EM")
    with pytest.raises(RuntimeError, match="非有限"):
        guarded_minimize(_Sim(False, False), "窗口 0 EM")
    guarded_minimize(_Sim(False, True), "窗口 0 EM")  # 正常路径不吭声


def test_guarded_parameter_helpers_name_the_parameter():
    ctx = _ParamContext({"ibs_bias_scale": 0.0})
    guarded_set_parameter(ctx, "ibs_bias_scale", 1.0, "窗口 0")
    assert guarded_get_parameter(ctx, "ibs_bias_scale", "窗口 0") == 1.0
    with pytest.raises(RuntimeError, match="ibs_f_0"):
        guarded_set_parameter(ctx, "ibs_f_0", 1.0, "窗口 0")
    with pytest.raises(RuntimeError, match="ibs_f_0"):
        guarded_get_parameter(ctx, "ibs_f_0", "窗口 0")


def test_require_parameters_is_the_only_way_to_catch_unregistered_symbols():
    """CustomCVForce 引用未注册符号不报错、静默给错数 —— 只能主动核对。"""
    ctx = _ParamContext({"cv_k_int": 0.0})
    require_parameters(ctx, ["cv_k_int"], "residual 力")
    with pytest.raises(RuntimeError, match="cv_k_rest"):
        require_parameters(ctx, ["cv_k_int", "cv_k_rest"], "residual 力")


def test_guarded_deserialize_reports_length_of_the_broken_cache():
    with pytest.raises(RuntimeError, match="反序列化失败（长度 8 字节）"):
        guarded_deserialize("<System>", "窗口 0 System 缓存")


def test_guarded_platform_lists_what_this_build_actually_has():
    with pytest.raises(RuntimeError) as ei:
        guarded_platform("NoSuchPlatform", "生产")
    assert "Reference" in str(ei.value), "错误信息必须列出这个 build 真有哪些平台"


def test_guarded_context_points_at_preexisting_vram_use():
    def _boom():
        raise RuntimeError("CUDA error: out of memory")
    with pytest.raises(RuntimeError, match="开跑前"):
        guarded_context(_boom, "REMD replica 3")
    assert guarded_context(lambda: "ctx", "正常") == "ctx"


class _NeedsSmallDt:
    """只有 dt <= dt_ok 才跑得动的窗口 —— 用来钉住"不要每段重爬一遍减半阶梯"。"""

    def __init__(self, dt_ok_ps, dt_ps=0.002):
        self.integrator = _FakeIntegrator(dt_ps)
        self.context = _FakeContext(self)
        self.dt_ok = dt_ok_ps
        self.finite = True
        self.calls = 0

    def step(self, n):
        self.calls += 1
        if self.integrator.dt.value_in_unit(unit.picoseconds) > self.dt_ok + 1e-12:
            raise openmm.OpenMMException("Particle coordinate is NaN")


def test_rollback_remembers_the_dt_this_window_needs():
    """真需要 0.5 fs 的窗口：只该在第一段付两次减半，后面 3 段直接过。

    早先每段成功都把 dt 弹回入口值，20 段就要白跑 60 段重试，看上去像卡死。
    """
    sim = _NeedsSmallDt(dt_ok_ps=0.0005, dt_ps=0.002)
    rescued = step_with_chunk_rollback(sim, 800, "t", chunk=200)
    assert rescued == 2, f"只该减半两次（2 fs→1 fs→0.5 fs），实际 {rescued}"
    assert sim.calls == 4 + 2, "4 段 + 2 次失败重试，不该每段重爬阶梯"
    assert sim.integrator.getStepSize().value_in_unit(unit.picoseconds) == pytest.approx(0.002)


def test_finite_check_says_which_quantity_blew_up():
    """坐标/力先炸而能量还有限时，why 不能回报一个正常的势能数字。"""
    class _Ctx:
        def getState(self, **kw):
            class S:
                def getPotentialEnergy(s_):
                    return -654321.0 * unit.kilojoule_per_mole
                def getPositions(s_, asNumpy=False):
                    p = np.zeros((3, 3)); p[1] = np.nan
                    return p * unit.nanometer
                def getForces(s_, asNumpy=False):
                    return np.zeros((3, 3)) * unit.kilojoule_per_mole / unit.nanometer
            return S()
    ok, why = finite_state_check(_Ctx())
    assert ok is False
    assert "坐标非有限" in why and "654321" not in why

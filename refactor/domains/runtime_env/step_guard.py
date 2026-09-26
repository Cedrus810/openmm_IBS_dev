"""OpenMM 积分步进的统一异常出口。

## 为什么值得单独一个模块

OpenMM 的 CUDA 平台是在 ``step()`` **内部**的原子重排里检查坐标、抛
``OpenMMException("Particle coordinate is NaN")`` 的 —— 不是等 step() 返回之后
由调用方查出来的。而本仓 30 来个 ``.step()`` 调用点里，绝大多数既没有 ``try``，
有限性检查也一律写在 step() 返回之后。于是真实的 NaN 一律以一条没有任何上下文
的 ``OpenMMException`` 炸穿整个窗口/整条腿：日志里看不出是哪个窗口、哪个阶段、
第几步，力分解诊断一次都打不出来，而写好的分段回退根本等不到触发的机会。

2026-09-09 的 OpenFF burn-in benchmark 就是这么丢掉 10 个 run 的：偏置预热的
``for ... sim.step(200)`` 循环里注释写着"每 200 步检查一次"，循环体里却什么都
没有，唯一的检查在整段 2000 步跑完之后。

这里只放两件事，其余模块一律从这里 import，不各写一份：

* :func:`guarded_step` —— 把 step() 的异常转成带上下文的 ``RuntimeError``。
  不吞异常，不改动力学。**生产/采样路径用它**：NaN 必须是一次干净的失败，
  不能靠偷偷改步长"救活"，那会悄悄换掉采样系综。
* :func:`step_with_chunk_rollback` —— 分段跑，炸了就回滚这一小段、步长减半
  重做。**只给热化/预热/松弛这类"还没进入生产态"的路径用。**

只依赖 openmm/numpy，因此 abfe_core / ibs_engine / abfe_preoptimizer /
abfe_pipeline 都能直接 import 而不产生循环依赖。

## 接入状态（2026-09-09）

**已接入**（计数核对于 2026-09-10）：``guarded_step``（ibs_engine **15** 处、
abfe_preoptimizer 10 处、abfe_pipeline 3 处）、``step_with_chunk_rollback``
（热化 / 偏置预热 / 死锁缓解松弛，ibs_engine 3 处）。

全仓 ``.step()`` 还剩 **11** 处裸调用（原文写 7 处，漏了
``outer_lambda_neural_basis`` 那一组）：

* ``ibs_engine.py:16890/17269``（生产 update 循环、余数补齐）、``21120``（REMD）
  —— 本来就在自己的 try 里；**生产采样段只能用 guarded_step**，
  绝不能换成会减半步长的 ``step_with_chunk_rollback``。
* ``abfe_preoptimizer.py:4170/4185``（2D 度量网格）—— 同上，已在 try 里。
* ``outer_lambda_neural_basis.py:2674/3096/4837`` —— 未接。
* ``free_energy_engine.py:1007/1014`` —— 另一条引擎，未接。

**下面这些全部尚未接入**，只是先把实现放好，等一次统一接线：

============================ ======== ============================================
guard                        裸调用点 典型失败（本仓真出过的）
============================ ======== ============================================
``guarded_minimize``               11 EM 自己在内部发散抛 NaN（2026-08-05 预热排序
                                      bug）、或返回后能量已是 inf（EXP-030 窗口 0
                                      EM 崩溃）。两个入口都要管：
                                      ``sim.minimizeEnergy`` 与
                                      ``LocalEnergyMinimizer.minimize(context)``。
``guarded_set_parameter``          71 名字打错/力没注册时 OpenMM 抛异常。
``require_parameters``              — 比抛异常更坏的一类：``CustomCVForce`` 引用
                                      未注册符号**不报错、静默给错数**
                                      （EXP-025 G4 Layer-1，差 2.27 kJ/mol）。
                                      这个只能主动核对，接不住。
``guarded_get_parameter``          14 读不存在的全局参数。
``guarded_deserialize``            42 resume 缓存被截断/写坏（NFS 上尤其）。
``guarded_platform``               34 CUDA 没编进这个 build、或名字拼错。
``guarded_context``                56 建 Context 时显存不够 —— 本仓真凶是别的库
                                      预分配整卡（P0 REMD：pymbar4 的 JAX 后端占
                                      75% 显存），报错文本跟"体系太大"分不开，
                                      所以要在信息里提示先看开跑前 used 多少。
============================ ======== ============================================

**不在这里做的**：checkpoint 跨 platform 迁移已经有
``abfe_pipeline.load_checkpoint_with_platform_migration``，不重写第二份。
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import openmm
from openmm import unit, XmlSerializer

__all__ = [
    "finite_state_check",
    "guarded_step",
    "step_with_chunk_rollback",
    "guarded_minimize",
    "guarded_set_parameter",
    "require_parameters",
    "guarded_get_parameter",
    "guarded_deserialize",
    "guarded_platform",
    "guarded_context",
]


def finite_state_check(context, *, check_forces: bool = True):
    """读一次 State，回答"这个 Context 现在是不是还是有限的"。

    返回 ``(ok, why)``；``ok=False`` 时 ``why`` 是一句可以直接打进日志的原因。
    永不抛：读 State 本身失败（Context 已经坏到读不出来）也算 ``ok=False``。

    单独抽出来是因为全仓有十几处各写一遍的"检查能量是不是 nan"，口径还不一致
    （有的只看能量、有的看能量+力、有的看坐标）。NaN 可以先出现在力或坐标上而
    能量还是有限的，所以默认三样都看。
    """
    try:
        st = context.getState(getEnergy=True, getForces=check_forces, getPositions=True)
        e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        bad = []
        if not np.isfinite(e):
            bad.append(f"势能={e!r} kJ/mol")
        n_bad_pos = int(np.count_nonzero(~np.isfinite(pos).all(axis=1)))
        if n_bad_pos:
            bad.append(f"{n_bad_pos} 个原子坐标非有限")
        if check_forces:
            f = st.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
            n_bad_f = int(np.count_nonzero(~np.isfinite(f).all(axis=1)))
            if n_bad_f:
                bad.append(f"{n_bad_f} 个原子受力非有限")
    except Exception as exc:  # noqa: BLE001 —— Context 坏到读不出来也是一种"不有限"
        return False, f"读取 State 失败：{type(exc).__name__}: {exc}"
    if bad:
        # 🔑 坐标/力先炸而能量还有限是常见组合（见上面 docstring）。这里必须报
        # **是哪一项**炸了；早先一律回报势能，于是日志会打出
        # "出现非有限值（势能=-654321.0 kJ/mol）"——读日志的人拿到的是反的信息。
        return False, "；".join(bad)
    return True, f"势能={e:.6g} kJ/mol"


def guarded_step(stepper, n_steps: int, label: str, *, on_error: Optional[Callable[[str], None]] = None) -> None:
    """跑 ``n_steps`` 步，把 OpenMM 抛的异常换成带上下文的 ``RuntimeError``。

    ``stepper`` 是任何带 ``.step(int)`` 的对象：``Simulation``、``Integrator``、
    ``context.getIntegrator()`` 都行。

    只改错误信息、不改动力学：该失败还是失败，只是失败时说得清是谁在哪一段炸的，
    并且（给了 ``on_error`` 时）先把诊断打出来。
    """
    try:
        stepper.step(int(n_steps))
    except openmm.OpenMMException as exc:
        if on_error is not None:
            try:
                on_error(label)
            except Exception as diag_exc:  # 诊断本身失败不能盖掉真正的异常
                print(f"    [WARN] {label}：诊断失败：{diag_exc}")
        raise RuntimeError(
            f"{label}：积分 {int(n_steps)} 步过程中出现非有限坐标/能量"
            f"（原始异常: {exc}）"
        ) from exc


def step_with_chunk_rollback(
    sim,
    total_steps: int,
    label: str,
    *,
    chunk: int = 500,
    min_dt_ps: float = 1.0e-5,
    on_exhausted: Optional[Callable[[str], None]] = None,
) -> int:
    """分段跑 ``total_steps`` 步，只回退炸掉的那一小段。

    每段前留一个回滚点（构型+速度）→ 跑这一段 → 检查能量/力/坐标是否有限；
    非有限就回滚该段、步长减半、重做，段过了恢复入口步长继续。连续减半到
    ``min_dt_ps`` 仍然炸才判失败（先调 ``on_exhausted(label)`` 打诊断，再抛
    ``RuntimeError``）。返回**回退重做的次数**（不是段数：同一段可能连续减半多次）。

    必须同时接住两种失败模式，缺一个这个回退就是死的：

    1. ``sim.step()`` **抛异常**（CUDA 平台的 NaN 检查在 step() 内部）。
    2. ``sim.step()`` 正常返回但能量/力/坐标已经是 inf/nan（静默污染）。

    2026-09-03 精简测试步进 schedule 时留下的那版回退只处理了第 2 种，而且只
    包住热化那一段，所以真实失败模式下它一次都没触发过。

    ⚠️ 只用于热化/预热/松弛。生产采样段请用 :func:`guarded_step` —— 在生产里
    偷偷减半步长等于换了积分器，会悄无声息地改掉系综。
    """
    entry_dt = sim.integrator.getStepSize()
    rescued = 0
    done = 0
    while done < total_steps:
        n = min(chunk, total_steps - done)
        # 只为这一段留一个回滚点，段过了就丢，不留全程快照。
        before = sim.context.getState(getPositions=True, getVelocities=True)
        dt_here = sim.integrator.getStepSize()
        while True:
            try:
                sim.step(n)
            except openmm.OpenMMException as exc:
                ok, why = False, f"积分器抛出 {type(exc).__name__}: {exc}"
            else:
                ok, why = finite_state_check(sim.context)
            if ok:
                break
            halved = dt_here.value_in_unit(unit.picoseconds) * 0.5
            if halved < min_dt_ps:
                print(f"    [ERR] {label}：步长已减半至下限仍出现非有限值（{why}），打印诊断：")
                if on_exhausted is not None:
                    try:
                        on_exhausted(label)
                    except Exception as diag_exc:
                        print(f"    [WARN] 诊断失败：{diag_exc}")
                sim.integrator.setStepSize(entry_dt)
                raise RuntimeError(
                    f"{label}：步长降到 {halved*1000:.3f} fs 仍出现非有限坐标/能量/力（{why}）"
                )
            # 回退这一段，减半重做
            sim.context.setState(before)
            dt_here = halved * unit.picoseconds
            sim.integrator.setStepSize(dt_here)
            rescued += 1
            print(
                f"    ↩️ {label}：第 {done}-{done+n} 步出现非有限值（{why}），"
                f"已回退该段，步长降至 {halved*1000:.2f} fs 重做"
            )
        # 🔑 [2026-09-09] 段成功后**保留**当前步长，只在整段跑完后恢复入口值。
        # 早先每段成功都弹回 entry_dt：真正需要 0.25 fs 的窗口，20 段里每一段都要
        # 从 2 fs 重新减半 3 次、每次白跑一整段，看上去就像卡死，而且它从不记住
        # "这个窗口需要多小的 dt"。代价是后续段的模拟时间变短——热化/预热是弛豫
        # 不是测量，这个代价换掉一个会被误读成 hang 的重试阶梯，是划算的。
        done += n
    sim.integrator.setStepSize(entry_dt)
    return rescued


# ---------------------------------------------------------------------------
# 以下全部**尚未接入**，见模块 docstring 的"接入状态"表。
# ---------------------------------------------------------------------------


def guarded_minimize(target, label: str, *, on_error: Optional[Callable[[str], None]] = None, **kwargs) -> None:
    """能量最小化的统一出口：抛异常要有上下文，"没抛但已经烂了"也要抓出来。

    ``target`` 可以是 ``Simulation``（走 ``minimizeEnergy``）也可以是 ``Context``
    （走 ``LocalEnergyMinimizer.minimize``），本仓两种入口都在用。``kwargs``
    原样透传（``maxIterations`` / ``tolerance``）。

    两种失败都要管，缺一个都不够：

    1. **EM 自己在内部发散**。2026-08-05 那个预热排序 bug 就是
       ``minimizeEnergy()`` 还没返回就抛 ``Particle coordinate is NaN``。
    2. **EM 正常返回，但结果已经是 inf/nan**。EXP-030 的窗口 0 EM 崩溃是这一类；
       只看"minimizeEnergy 有没有抛"的检查全程无感。
    """
    context = getattr(target, "context", target)
    try:
        if hasattr(target, "minimizeEnergy"):
            target.minimizeEnergy(**kwargs)
        else:
            openmm.LocalEnergyMinimizer.minimize(target, **kwargs)
    except openmm.OpenMMException as exc:
        if on_error is not None:
            try:
                on_error(label)
            except Exception as diag_exc:
                print(f"    [WARN] {label}：诊断失败：{diag_exc}")
        raise RuntimeError(f"{label}：能量最小化过程中发散（原始异常: {exc}）") from exc
    ok, why = finite_state_check(context)
    if not ok:
        if on_error is not None:
            try:
                on_error(label)
            except Exception as diag_exc:
                print(f"    [WARN] {label}：诊断失败：{diag_exc}")
        raise RuntimeError(f"{label}：能量最小化返回了非有限的能量/力/坐标（{why}）")


def guarded_set_parameter(context, name: str, value: float, label: str) -> None:
    """写全局参数。名字不存在时 OpenMM 抛的异常不带调用现场，这里补上。"""
    try:
        context.setParameter(str(name), float(value))
    except openmm.OpenMMException as exc:
        raise RuntimeError(
            f"{label}：设置全局参数 {name}={value!r} 失败 —— 该参数在这个 Context 里"
            f"不存在（力没建、名字拼错、或前缀不对）。原始异常: {exc}"
        ) from exc


def guarded_get_parameter(context, name: str, label: str) -> float:
    """读全局参数，读不到时说清是哪个名字、谁在读。"""
    try:
        return float(context.getParameter(str(name)))
    except openmm.OpenMMException as exc:
        raise RuntimeError(
            f"{label}：读取全局参数 {name} 失败 —— 该参数在这个 Context 里不存在。"
            f"原始异常: {exc}"
        ) from exc


def require_parameters(context, names, label: str) -> None:
    """**主动核对**一批全局参数确实都在 —— 这一类失败接不住，只能查。

    ``CustomCVForce`` 引用一个没被注册的符号时，OpenMM **不报错**，安安静静给出
    错误的能量（EXP-025 G4 Layer-1：``OuterLambdaResidualBiasForce`` 没注册
    ``cv_k_int``/``cv_k_rest``，结果差 2.27 kJ/mol，靠"每个输入独立验证"才定位到）。
    没有任何 try/except 能救这种，所以凡是靠外部注册符号的力，建好之后应当在这里
    过一遍。
    """
    missing = []
    for n in names:
        try:
            context.getParameter(str(n))
        except openmm.OpenMMException:
            missing.append(str(n))
    if missing:
        raise RuntimeError(
            f"{label}：Context 缺少必需的全局参数 {missing} —— 引用未注册符号的力"
            "不会报错，只会静默给出错误能量，因此这里 fail-closed。"
        )


def guarded_deserialize(xml_text: str, label: str):
    """``XmlSerializer.deserialize`` 的出口：缓存被截断/写坏时说清是哪个缓存。

    resume 缓存写在 NFS 上，半截文件不是假想情况。原始异常只会说 XML 解析失败，
    不会说是哪一份。
    """
    try:
        return XmlSerializer.deserialize(xml_text)
    except Exception as exc:  # OpenMMException / ValueError / ExpatError 都可能
        n = len(xml_text) if xml_text is not None else 0
        raise RuntimeError(
            f"{label}：反序列化失败（长度 {n} 字节）—— 缓存可能被截断或写坏，"
            f"应当重建而不是继续用。原始异常: {type(exc).__name__}: {exc}"
        ) from exc


def guarded_platform(name: str, label: str):
    """取 Platform。名字拼错、或这个 OpenMM build 根本没编该平台时给人话。"""
    try:
        return openmm.Platform.getPlatformByName(str(name))
    except Exception as exc:
        available = [
            openmm.Platform.getPlatform(i).getName()
            for i in range(openmm.Platform.getNumPlatforms())
        ]
        raise RuntimeError(
            f"{label}：取不到 Platform {name!r} —— 这个 OpenMM build 只有 {available}。"
            f"原始异常: {exc}"
        ) from exc


def guarded_context(builder: Callable[[], object], label: str):
    """建 Context/Simulation 的出口，专治"显存不够"被报成别的东西。

    ``builder`` 是一个无参可调用，返回 Context 或 Simulation。

    本仓真出过的 P0：pymbar4 的 JAX 后端一 import 就预分配整卡 75% 显存，之后
    REMD 建 Context 直接失败，报错文本跟"体系太大装不下"长得一模一样，查了很久。
    所以这里把"先看开跑前 used 是多少"写进错误信息里，别再靠人记。
    """
    try:
        return builder()
    except Exception as exc:
        text = str(exc).lower()
        hint = ""
        if "memory" in text or "cuda" in text or "out of" in text:
            hint = (
                " —— 若是显存问题，先确认**开跑前**卡上已经被占了多少"
                "（别的库可能预分配了整卡，例如 pymbar4 的 JAX 后端占 75%），"
                "再判断是不是体系本身太大。"
            )
        raise RuntimeError(
            f"{label}：建立 Context 失败{hint} 原始异常: {type(exc).__name__}: {exc}"
        ) from exc

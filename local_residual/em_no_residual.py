"""Process-local EM policy for the optional LocalManyBodyResidual sampler.

The candidate System keeps the real residual Hamiltonian.  Only the cold-start
minimization is redirected to a residual-free twin, then positions and box
vectors are copied back.  The patch is deliberately generic and lives in the
mainline package; experiment launchers may import it as a compatibility shim.
"""
from __future__ import annotations

import hashlib

_STASH: dict = {}
_INSTALLED = False
_ORIGINAL_BUILD_WINDOW_SYSTEM = None
_ORIGINAL_MINIMIZE = None
_IBS_ENGINE = None
_APP = None


def _system_sha256(system, openmm) -> str:
    return hashlib.sha256(openmm.XmlSerializer.serialize(system).encode("utf-8")).hexdigest()


def uninstall() -> None:
    """Restore both patched methods and discard a pending twin, idempotently."""
    global _INSTALLED, _ORIGINAL_BUILD_WINDOW_SYSTEM, _ORIGINAL_MINIMIZE
    global _IBS_ENGINE, _APP
    if _INSTALLED:
        if _IBS_ENGINE is not None and _ORIGINAL_BUILD_WINDOW_SYSTEM is not None:
            _IBS_ENGINE.IBSWindowManagerDualLambda._build_window_system = (
                _ORIGINAL_BUILD_WINDOW_SYSTEM
            )
        if _APP is not None and _ORIGINAL_MINIMIZE is not None:
            _APP.Simulation.minimizeEnergy = _ORIGINAL_MINIMIZE
    _STASH.clear()
    _INSTALLED = False
    _ORIGINAL_BUILD_WINDOW_SYSTEM = None
    _ORIGINAL_MINIMIZE = None
    _IBS_ENGINE = None
    _APP = None


def install() -> None:
    """Install the twin policy once for the current Python process."""
    global _INSTALLED, _ORIGINAL_BUILD_WINDOW_SYSTEM, _ORIGINAL_MINIMIZE
    global _IBS_ENGINE, _APP
    if _INSTALLED:
        return
    import openmm
    from openmm import app, unit
    import ibs_engine

    original_build = ibs_engine.IBSWindowManagerDualLambda._build_window_system
    original_minimize = app.Simulation.minimizeEnergy
    _ORIGINAL_BUILD_WINDOW_SYSTEM = original_build
    _ORIGINAL_MINIMIZE = original_minimize
    _IBS_ENGINE = ibs_engine
    _APP = app

    def _find_global_parameter_suffix(system, suffix: str):
        for force_index in range(system.getNumForces()):
            force = system.getForce(force_index)
            if not isinstance(force, openmm.CustomCVForce):
                continue
            for parameter_index in range(force.getNumGlobalParameters()):
                name = force.getGlobalParameterName(parameter_index)
                if name.endswith(suffix):
                    return name
        return None

    def _system_has_residual_cv(system) -> bool:
        for force_index in range(system.getNumForces()):
            force = system.getForce(force_index)
            if not isinstance(force, openmm.CustomCVForce):
                continue
            for cv_index in range(force.getNumCollectiveVariables()):
                if force.getCollectiveVariableName(cv_index) == "exp025_residual_basis":
                    return True
        return False

    def patched_build_window_system(self, lc_win, lv_win, resolved_box, positions):
        _STASH.clear()
        real_system, ibs_wrap = original_build(self, lc_win, lv_win, resolved_box, positions)
        if not getattr(ibs_wrap, "residual_enabled", False):
            return real_system, ibs_wrap

        # 🔑 [2026-09-09] 孪生 System 改成**惰性**构造。
        #
        # 原来在这里就跑第二遍 `original_build` 把整份孪生 System 造出来塞进
        # `_STASH`。于是任何"窗口建好了但没走到配对的 minimizeEnergy"的情形
        # （resume 跳过 EM、中途异常、或下面 SHA 不匹配那条 return 分支）
        # 都会让一份完整 System 一直挂在模块全局里直到进程结束。
        # 改成只 stash 一个无参可调用；真正需要时（patched_minimize 里）才构造，
        # 用完即释放那份 System。行为不变：构造参数与时机的语义完全一致。
        #
        # [2026-09-10] 可调用本身**跨多次 minimize 保留**（见 patched_minimize 的
        # finally），因为同一个窗口里不止一次原生最小化。下面这行 `_STASH.clear()`
        # 是它唯一的生命周期边界：一个窗口一份。
        def _make_em_system():
            saved_factory = self.residual_basis_force_factory
            self.residual_basis_force_factory = None
            try:
                em_system, _em_wrap = original_build(
                    self, lc_win, lv_win, resolved_box, positions
                )
            finally:
                self.residual_basis_force_factory = saved_factory
            return em_system

        _STASH.update(
            make_em_system=_make_em_system,
            real_system_sha256=_system_sha256(real_system, openmm),
            topology=self.topology,
            temperature=self.temperature,
        )
        print(
            "  🧪 [EM-no-residual] 候选窗口已登记不含 LocalManyBodyResidualForce 的"
            "孪生 System 构造器；仅在冷启动最小化时才真正建出来。",
            flush=True,
        )
        return real_system, ibs_wrap

    def patched_minimize(
        self,
        tolerance=10 * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=0,
        reporter=None,
    ):
        make_em_system = _STASH.get("make_em_system")
        if make_em_system is None:
            return original_minimize(
                self, tolerance=tolerance, maxIterations=maxIterations, reporter=reporter
            )

        expected_sha = _STASH.get("real_system_sha256")
        real_system = self.context.getSystem()
        if expected_sha != _system_sha256(real_system, openmm):
            # Fixed-state endpoint/path probes intentionally derive a
            # residual-free System from the candidate window's common XML.
            # Their native EM is safe and must not consume the pending twin;
            # a changed System that still contains the residual CV is the
            # unsafe case and fails closed.
            if not _system_has_residual_cv(real_system):
                return original_minimize(
                    self,
                    tolerance=tolerance,
                    maxIterations=maxIterations,
                    reporter=reporter,
                )
            _STASH.clear()
            raise RuntimeError(
                "[EM-no-residual] twin 与当前含残差 Simulation System 不匹配；"
                "拒绝原生最小化"
            )

        temp_sim = None
        em_system = None
        try:
            em_system = make_em_system()
            state = self.context.getState(getPositions=True)
            box = state.getPeriodicBoxVectors()
            platform = self.context.getPlatform()
            properties = {}
            for name in (
                "CudaPrecision", "CudaDeviceIndex", "CudaUseBlockingSync",
                "CudaCompiler", "CudaTempDirectory",
            ):
                try:
                    properties[name] = platform.getPropertyValue(self.context, name)
                except Exception:
                    pass
            # 🔑 [2026-09-09] 精度属性抓不到就 fail closed。
            #
            # 上面是逐项 best-effort（不同 platform 的属性名集合不同，抓不到很正常），
            # 但 `CudaPrecision` 是例外：抓不到就不传 ⇒ 孪生 Simulation 用 CUDA
            # **默认单精度**做最小化，再把坐标拷回 mixed 精度的生产 Context，
            # EM 结果与生产 Hamiltonian 的精度约定不一致，而且没有任何日志。
            # 本仓的真实生产就是 CUDA mixed（见 EXP-025 的 mixed precision 支持），
            # 所以这条静默降级是会真的发生的。
            if str(platform.getName()).upper() == "CUDA" and "CudaPrecision" not in properties:
                raise RuntimeError(
                    "[EM-no-residual] 读不到生产 Context 的 CudaPrecision，"
                    "无法保证孪生 Simulation 与生产同精度。"
                    "静默落回 CUDA 默认单精度会让 EM 结果与生产 Hamiltonian 的"
                    "精度约定不一致，拒绝继续（fail closed）。"
                )
            temp_integrator = openmm.LangevinMiddleIntegrator(
                _STASH.get("temperature", 300.0 * unit.kelvin),
                2.0 / unit.picosecond,
                0.002 * unit.picosecond,
            )
            temp_integrator.setConstraintTolerance(self.integrator.getConstraintTolerance())
            temp_sim = app.Simulation(
                _STASH["topology"], em_system, temp_integrator, platform, properties
            )
            if box is not None:
                temp_sim.context.setPeriodicBoxVectors(*box)
            temp_sim.context.setPositions(state.getPositions())
            print(
                "  🧪 [EM-no-residual] 在残差-free twin 上执行最小化；"
                "LocalManyBodyResidualForce 不参与本次 EM。",
                flush=True,
            )
            original_minimize(
                temp_sim, tolerance=tolerance, maxIterations=maxIterations, reporter=reporter
            )
            minimized = temp_sim.context.getState(getPositions=True)
            self.context.setPositions(minimized.getPositions())
            minimized_box = minimized.getPeriodicBoxVectors()
            if minimized_box is not None:
                self.context.setPeriodicBoxVectors(*minimized_box)
            # The production state machine restores the residual Hamiltonian
            # before warmup.  Zeroing these two globals closes the tiny
            # post-EM diagnostic-step gap before that explicit restore.
            # 🔑 [2026-09-09] 找不到就出声。`_find_global_parameter_suffix` 只扫
            # `CustomCVForce` 的 global parameters，一旦参数挂在别的力上、或者前缀/
            # 命名变了，它返回 None 而这里**什么都不做** —— 上面注释里说要关掉的
            # "post-EM 诊断步残差窗口"实际没关，是 fail-open。至少要让它可见。
            for suffix in ("_bias_scale", "_s_residual"):
                name = _find_global_parameter_suffix(real_system, suffix)
                if name is None:
                    print(
                        f"  [WARN] [EM-no-residual] 在 CustomCVForce 上找不到以 "
                        f"{suffix!r} 结尾的全局参数，post-EM 的残差窗口**没有**被关掉。"
                        "若参数改挂到了别的力上，需要同步改 _find_global_parameter_suffix。",
                        flush=True,
                    )
                    continue
                self.context.setParameter(name, 0.0)
        finally:
            # 🔑 [2026-09-10] **不再** `_STASH.clear()`。
            #
            # 清掉它等于让这套策略变成"每个窗口只保护第一次 minimizeEnergy"。
            # 同一个窗口后面还有两次原生最小化会被静默放过：
            #   * `ibs_engine.py` 的 `_production_disaster_rollback`
            #     → `sim.minimizeEnergy(maxIterations=2000, tolerance=1.0)`
            #   * 约束死锁缓解分支
            #     → `sim.minimizeEnergy(maxIterations=5000, tolerance=10.0)`
            # 两者都会命中 `make_em_system is None` 直接走 original_minimize，
            # **带着 LocalManyBodyResidualForce 跑 EM** —— 正是这套补丁存在的
            # 唯一目的，而且一行日志都没有。实测复现过（twin built count 停在 1）。
            #
            # 保留 stash 是安全的：上面 `real_system_sha256` 那道门会核对当前
            # Context 的 System 与登记时是否同一份，不匹配才走原生路径。
            # 生命周期仍然有界 —— `patched_build_window_system` 每次进来都
            # 无条件重设 `_STASH`，所以它最多活到下一个窗口建起来为止。
            if temp_sim is not None:
                try:
                    del temp_sim.context
                except Exception:
                    pass
                del temp_sim
            # 孪生 System 与生产 Context 在 EM 期间必然同时驻留（要从生产 Context
            # 读坐标、再写回去），这是这套做法固有的；但它不该活过 EM。
            em_system = None
            import gc as _gc
            _gc.collect()

    ibs_engine.IBSWindowManagerDualLambda._build_window_system = patched_build_window_system
    app.Simulation.minimizeEnergy = patched_minimize
    _INSTALLED = True
    print("  🧪 [EM-no-residual] 已安装当前进程 twin EM 策略。", flush=True)

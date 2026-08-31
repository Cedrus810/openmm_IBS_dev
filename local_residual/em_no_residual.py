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

        saved_factory = self.residual_basis_force_factory
        self.residual_basis_force_factory = None
        try:
            em_system, _em_wrap = original_build(self, lc_win, lv_win, resolved_box, positions)
        finally:
            self.residual_basis_force_factory = saved_factory
        _STASH.update(
            em_system=em_system,
            real_system_sha256=_system_sha256(real_system, openmm),
            topology=self.topology,
            temperature=self.temperature,
        )
        print(
            "  🧪 [EM-no-residual] 已为候选窗口建立不含 LocalManyBodyResidualForce 的"
            "孪生 System；仅用于冷启动最小化。",
            flush=True,
        )
        return real_system, ibs_wrap

    def patched_minimize(
        self,
        tolerance=10 * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=0,
        reporter=None,
    ):
        em_system = _STASH.get("em_system")
        if em_system is None:
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
        try:
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
            for suffix in ("_bias_scale", "_s_residual"):
                name = _find_global_parameter_suffix(real_system, suffix)
                if name is not None:
                    self.context.setParameter(name, 0.0)
        finally:
            _STASH.clear()
            if temp_sim is not None:
                try:
                    del temp_sim.context
                except Exception:
                    pass
                del temp_sim

    ibs_engine.IBSWindowManagerDualLambda._build_window_system = patched_build_window_system
    app.Simulation.minimizeEnergy = patched_minimize
    _INSTALLED = True
    print("  🧪 [EM-no-residual] 已安装当前进程 twin EM 策略。", flush=True)

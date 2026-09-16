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
_ORIGINAL_SET_PARAMETER = None
_ORIGINAL_GET_STATE = None
_IBS_ENGINE = None
_APP = None
_OPENMM = None

# IBS 偏置力（`CustomCVForce`）的力组。残差 CV 就挂在它里面。
_IBS_BIAS_FORCE_GROUP = 1
# CustomCVForce 里残差那个 CV 的名字（`local_residual.openmm_plugin` 建的就是它）。
_RESIDUAL_CV_NAME = "exp025_residual_basis"
# `id(System) -> bool`。`getState`/`setParameter` 是热路径，不能每次都扫一遍力。
# `uninstall()` 清空；System 与 Context 同生命周期，窗口换了就是新对象。
_RESIDUAL_CV_CACHE: dict = {}


def _system_sha256(system, openmm) -> str:
    return hashlib.sha256(openmm.XmlSerializer.serialize(system).encode("utf-8")).hexdigest()


def _has_residual_cv(system) -> bool:
    """这个 System 里挂着残差 CV 吗（带缓存）。baseline 恒为 False。"""
    key = id(system)
    hit = _RESIDUAL_CV_CACHE.get(key)
    if hit is not None:
        return hit
    found = False
    if _OPENMM is not None:
        for force in system.getForces():
            if not isinstance(force, _OPENMM.CustomCVForce):
                continue
            for i in range(force.getNumCollectiveVariables()):
                if force.getCollectiveVariableName(i) == _RESIDUAL_CV_NAME:
                    found = True
                    break
            if found:
                break
    _RESIDUAL_CV_CACHE[key] = found
    return found


def _sync_integration_force_groups(context, bias_scale: float) -> None:
    """🔑🔑 [2026-09-16 / docs/TODO.md LR-06] `bias_scale == 0` ⟹ 把 Group-1 移出积分力组。

    `bias_scale` 乘在 Group-1 **整个**表达式外面，而里面就是配体↔环境的软核
    相互作用 ⟹ 它为 0 不是"关掉偏置"，是**把配体关成完全的鬼影**：水分子直接穿过
    配体，`r → 0` 成为必然。而 `CustomCVForce` 会求值它的**每一个** CV（与系数是否
    为 0 无关）⟹ `LocalManyBodyResidualForce` 在这些几何上撞 0.1 Å 硬门，整条 run
    死在窗口 0 热化。把力组排除掉，插件在鬼影期就根本不被求值。

    ⚠️ **只对挂了残差 CV 的 System 生效。** 理论上 `bias_scale=0` 时 Group-1 贡献是
    精确的零、排不排除等价；实测在真实窗口 0 System 上（CUDA mixed precision）
    ΔE=7.5e-5 kJ/mol、max|ΔF|=1.8e-4（相对 1e-8，是归约顺序的浮点噪声）。MD 是混沌的，
    1e-8 也会让轨迹分叉，而 `bias_scale=0` 在 **baseline 也会发生**（dt 爬坡全程、
    偏置爬坡起点）⟹ 一律排除会改掉 baseline 的预热轨迹。baseline 的 System 没有残差
    CV，这里直接返回，连 `setIntegrationForceGroups` 都不调。
    """
    system = context.getSystem()
    if not _has_residual_cv(system):
        return
    integrator = context.getIntegrator()
    setter = getattr(integrator, "setIntegrationForceGroups", None)
    if setter is None:            # 老 OpenMM 没这个 API：退回改动前行为
        return
    mask = 0
    for force in system.getForces():
        group = int(force.getForceGroup())
        if bias_scale == 0.0 and group == _IBS_BIAS_FORCE_GROUP:
            continue
        mask |= 1 << group
    setter(mask)


def uninstall() -> None:
    """Restore both patched methods and discard a pending twin, idempotently."""
    global _INSTALLED, _ORIGINAL_BUILD_WINDOW_SYSTEM, _ORIGINAL_MINIMIZE
    global _ORIGINAL_SET_PARAMETER, _ORIGINAL_GET_STATE
    global _IBS_ENGINE, _APP, _OPENMM
    if _INSTALLED:
        if _IBS_ENGINE is not None and _ORIGINAL_BUILD_WINDOW_SYSTEM is not None:
            _IBS_ENGINE.IBSWindowManagerDualLambda._build_window_system = (
                _ORIGINAL_BUILD_WINDOW_SYSTEM
            )
        if _APP is not None and _ORIGINAL_MINIMIZE is not None:
            _APP.Simulation.minimizeEnergy = _ORIGINAL_MINIMIZE
        if _OPENMM is not None and _ORIGINAL_SET_PARAMETER is not None:
            _OPENMM.Context.setParameter = _ORIGINAL_SET_PARAMETER
        if _OPENMM is not None and _ORIGINAL_GET_STATE is not None:
            _OPENMM.Context.getState = _ORIGINAL_GET_STATE
    _STASH.clear()
    _RESIDUAL_CV_CACHE.clear()
    _INSTALLED = False
    _ORIGINAL_BUILD_WINDOW_SYSTEM = None
    _ORIGINAL_MINIMIZE = None
    _ORIGINAL_SET_PARAMETER = None
    _ORIGINAL_GET_STATE = None
    _IBS_ENGINE = None
    _APP = None
    _OPENMM = None


def install() -> None:
    """Install the twin policy once for the current Python process."""
    global _INSTALLED, _ORIGINAL_BUILD_WINDOW_SYSTEM, _ORIGINAL_MINIMIZE
    global _ORIGINAL_SET_PARAMETER, _ORIGINAL_GET_STATE
    global _IBS_ENGINE, _APP, _OPENMM
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
    _OPENMM = openmm
    _RESIDUAL_CV_CACHE.clear()

    # ---- [2026-09-16 / LR-06] 预热鬼影期不求值残差 CV：两个 Context 级补丁 ----
    #
    # 为什么做成猴子补丁而不是改主线：这套东西还没经过大规模验证，主线
    # （`ibs_engine` / `step_guard`）保持零改动，验证完再谈合并。整套补丁只在
    # `--outer-lambda-local-residual-ibs` 开启时由
    # `runabfe._scope_pipeline_with_optional_outer_lambda_em` 安装，baseline 不装。
    original_set_parameter = openmm.Context.setParameter
    original_get_state = openmm.Context.getState
    _ORIGINAL_SET_PARAMETER = original_set_parameter
    _ORIGINAL_GET_STATE = original_get_state

    def patched_set_parameter(self, name, value):
        """任何一处改 `*_bias_scale` 都顺带同步积分力组。

        补丁挂在 `Context.setParameter` 上而不是某个具名 helper，是因为主线里改
        `bias_scale` 的地方有 **8 处**（热化前后、dt 爬坡、偏置爬坡的每一档、
        resume 的几条分支）—— 逐个去包必然漏，漏掉的那一处就又是一个鬼影期里
        带着残差 CV 空转的窗口。
        """
        original_set_parameter(self, name, value)
        if not str(name).endswith("_bias_scale"):
            return
        try:
            _sync_integration_force_groups(self, float(value))
        except Exception as exc:  # noqa: BLE001 —— 同步失败不能盖掉调用方的赋值
            print(f"  [WARN] [EM-no-residual] 同步积分力组失败：{exc}", flush=True)

    def patched_get_state(self, *args, **kwargs):
        """没显式给 `groups` 时，按积分器**实际在积分**的力组取 State。

        只在积分力组被限制过（= 我们的鬼影期）且该 System 挂了残差 CV 时才介入；
        其余情况 `getIntegrationForceGroups()` 返回 -1，逐位走原路径。

        必要性：把 Group-1 移出积分**只管积分**。`step_guard.finite_state_check`
        每 500 步会做一次不带 `groups` 的 `getState(getEnergy/Forces/Positions)`，
        那会把 Group-1 连同残差 CV 一起求值 ⟹ 插件照样在鬼影几何上抛异常 ⟹
        被当成"状态非有限"，回退 + 步长减半一路走到窗口失败。
        """
        # `groups` 是 getState 的第 10 个参数；位置参数少于 10 个就说明没给它。
        if "groups" not in kwargs and len(args) < 10:
            try:
                mask = self.getIntegrator().getIntegrationForceGroups()
                if (isinstance(mask, int) and mask >= 0
                        and _has_residual_cv(self.getSystem())):
                    kwargs["groups"] = mask
            except Exception:  # noqa: BLE001 —— 取不到就退回全量
                pass
        return original_get_state(self, *args, **kwargs)

    openmm.Context.setParameter = patched_set_parameter
    openmm.Context.getState = patched_get_state

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
            # 🔑🔑 [2026-09-16] **只关 `_s_residual`，绝不碰 `_bias_scale`。**
            #
            # `bias_scale` 乘在 Group-1 **整个**表达式外面（`ibs_engine` 里
            # `f"{prefix}_bias_scale * ({_state_expr(0)} - kt*(...))"`），而
            # `_state_expr(0)` 里就是 `cv_0_int + cv_0_rest` —— **配体↔环境的软核
            # 相互作用**。把它清零不是"关掉偏置"，是**把配体关成完全的鬼影**：水分子
            # 直接穿过配体，`r → 0` 成为必然，而 `CustomCVForce` 仍会逐步求值它的
            # 每一个 CV（与系数是否为 0 无关）⟹ 插件在这些几何上撞 0.1 Å 硬门，
            # 整条 run 死在窗口 0 热化。真机 cyclod_ligand1_outer 三个 rep 全中，
            # 本机复现 3/3；拦掉这一次清零后热化**完全通过**（docs/TODO.md `LR-06`）。
            #
            # `ibs_engine` 自己在最小化前只关 `s_residual`，注释明写「不像
            # bias_scale=0 那样连 baseline 也在正常使用的物理 softcore-state 混合力
            # 一起关掉」—— 这里原来的两元组与那条约定直接冲突，且因为在它之后执行
            # 而赢了。附带后果：baseline 不装本补丁 ⟹ 两臂预热的哈密顿量不同。
            #
            # 恢复点在 `ibs_engine` 那句 `setParameter(f"{prefix}_s_residual", 1.0)`。
            for suffix in ("_s_residual",):
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

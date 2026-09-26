# ============================================================================
# 2. IBS 偏置力与采样器 (修复版)
# ============================================================================

class IBSBiasForce:
    """IBS 偏置力封装 (Group 1) - 数值稳定差分形式

    EXP-025 G4 Layer-2 (2026-08-13): optional, disabled-by-default native
    shared residual basis. When ``residual_basis_force`` is None (the
    default), this class's behavior is BYTE-IDENTICAL to before this
    addition -- same expression string, same CVs, same globals, same force
    group. This is deliberate and load-bearing: production callers that
    never pass the new keyword-only arguments must see zero behavior
    change, verified by an XML-diff regression test (see
    scripts/test_exp025_g4_ibsbiasforce_native_residual.py), not just
    argued informally.

    When enabled, each state's discriminant argument becomes
    ``X_k = cv_k_int + cv_k_rest + A_k*(exp025_residual_basis - U_offset) - f_k``
    with the shared basis Force registered as exactly ONE collective
    variable (never duplicated per state) inside the SAME Group-1
    CustomCVForce -- the plugin executes once per force evaluation, and the
    max-pivot log-sum-exp discriminant is otherwise unchanged.
    """

    @classmethod
    def with_residual_basis(
        cls,
        n_states: int,
        temperature: openmm.unit.Quantity,
        prefix: str = "abfe",
        *,
        residual_basis_force: Optional[openmm.Force] = None,
        residual_state_coefficients: Optional[Sequence[float]] = None,
        residual_energy_offset_kj_mol: float = 0.0,
    ) -> "IBSBiasForce":
        """Construct the opt-in residual variant without changing the legacy API."""
        instance = cls.__new__(cls)
        instance._residual_constructor_config = {
            "residual_basis_force": residual_basis_force,
            "residual_state_coefficients": residual_state_coefficients,
            "residual_energy_offset_kj_mol": residual_energy_offset_kj_mol,
        }
        cls.__init__(instance, n_states, temperature, prefix)
        return instance

    def __init__(
        self,
        n_states: int,
        temperature: openmm.unit.Quantity,
        prefix: str = "abfe",
        **residual_kwargs,
    ):
        residual_config = getattr(self, "_residual_constructor_config", None)
        if residual_kwargs:
            allowed_residual_kwargs = {
                "residual_basis_force",
                "residual_state_coefficients",
                "residual_energy_offset_kj_mol",
            }
            unexpected = sorted(set(residual_kwargs) - allowed_residual_kwargs)
            if unexpected:
                raise TypeError(
                    "IBSBiasForce.__init__() got unexpected keyword argument(s): "
                    f"{', '.join(unexpected)}"
                )
            if residual_config is not None:
                raise TypeError(
                    "IBSBiasForce: residual configuration was supplied both through "
                    "with_residual_basis() and direct constructor keywords"
                )
            # Preserve the EXP-025 direct keyword path for existing callers. New
            # production wiring uses the named classmethod, while this path keeps
            # older scripts and downstream integrations behaviorally unchanged.
            residual_config = {
                "residual_basis_force": residual_kwargs.get("residual_basis_force"),
                "residual_state_coefficients": residual_kwargs.get("residual_state_coefficients"),
                "residual_energy_offset_kj_mol": residual_kwargs.get(
                    "residual_energy_offset_kj_mol", 0.0
                ),
            }
        if residual_config is None:
            residual_basis_force = None
            residual_state_coefficients = None
            residual_energy_offset_kj_mol = 0.0
        else:
            residual_basis_force = residual_config["residual_basis_force"]
            residual_state_coefficients = residual_config[
                "residual_state_coefficients"
            ]
            residual_energy_offset_kj_mol = residual_config[
                "residual_energy_offset_kj_mol"
            ]
        self.n_states = n_states
        self.prefix = prefix
        self._cv_keeper = []
        self._int_cv_force_xmls = []
        # [性能修复：主 Context 一次读全部 CV] 与 _int_cv_force_xmls 平行的索引表——
        # 每个 "_int" CV 在主 CustomCVForce (self.force) 里的真实注册索引，供
        # IBSSampler._evaluate_interaction_energies_live() 用
        # getCollectiveVariableValues() 一次性取全部值，不必重建 probe System。
        # [EXP-031 S1] 形态在**构造时**冻结到实例上，之后一切按 self.omit_zero_rest_cvs
        # 走：注册点、_state_expr、validate_wiring、CV 预算检查必须看同一个值，否则
        # 中途翻模块开关会造出"表达式引用了没注册的 CV"这种静默给错数的 Force
        # （EXP-025 G4 Layer-1 就是这个失败模式）。
        self.omit_zero_rest_cvs = bool(IBS_BIAS_OMIT_ZERO_REST_CVS)
        self._int_cv_indices: List[int] = []
        self._residual_basis_cv_index: Optional[int] = None
        self.residual_enabled = residual_basis_force is not None
        if not self.residual_enabled:
            if residual_state_coefficients is not None:
                raise ValueError(
                    "IBSBiasForce: residual_state_coefficients must be None when residual_basis_force is None "
                    "(both or neither -- see class docstring)"
                )
            if residual_energy_offset_kj_mol != 0.0:
                raise ValueError(
                    "IBSBiasForce: residual_energy_offset_kj_mol must be 0.0 when residual_basis_force is None "
                    "(both or neither -- see class docstring)"
                )
        else:
            if residual_state_coefficients is None or len(residual_state_coefficients) != n_states:
                raise ValueError(
                    f"IBSBiasForce: residual_state_coefficients must have exactly n_states={n_states} entries "
                    f"when residual_basis_force is provided (got "
                    f"{None if residual_state_coefficients is None else len(residual_state_coefficients)})"
                )
            if not all(math.isfinite(float(c)) for c in residual_state_coefficients):
                raise ValueError("IBSBiasForce: residual_state_coefficients must all be finite")
            if not math.isfinite(float(residual_energy_offset_kj_mol)):
                raise ValueError("IBSBiasForce: residual_energy_offset_kj_mol must be finite")
            # Fail closed on the CV budget HERE, before any addCollectiveVariable()
            # call -- do not rely on OpenMM to reject an over-budget CustomCVForce
            # at some later, less predictable point (matches the EXP-025 G4
            # Layer-1 lesson: a wiring contract violation must be caught by an
            # explicit check, not "it didn't throw so it must be fine").
            # [EXP-031 S1] per-state CV count follows the Group-1 form: 1 when the
            # identically-zero cv_k_rest placeholders are omitted, 2 otherwise. The
            # "+1" is the single shared exp025_residual_basis CV either way.
            cvs_per_state = ibs_dual_cvs_per_lambda_state(self.omit_zero_rest_cvs)
            required_cv_count = cvs_per_state * n_states + 1
            if required_cv_count > OPENMM_CUSTOM_CV_MAX_VARIABLES:
                raise ValueError(
                    f"IBSBiasForce: enabling the native residual basis needs {required_cv_count} collective "
                    f"variables ({cvs_per_state}*n_states+1) for n_states={n_states}, exceeding the hard "
                    f"ceiling of {OPENMM_CUSTOM_CV_MAX_VARIABLES} -- i.e. n_states must be <= "
                    f"{(OPENMM_CUSTOM_CV_MAX_VARIABLES - 1) // cvs_per_state} with the residual basis enabled"
                )
        self.residual_state_coefficients = (
            [float(c) for c in residual_state_coefficients] if self.residual_enabled else None
        )
        self.residual_energy_offset_kj_mol = float(residual_energy_offset_kj_mol)
        if isinstance(temperature, float):
            temperature = temperature * openmm.unit.kelvin
        kt = (unit.MOLAR_GAS_CONSTANT_R * temperature).value_in_unit(openmm.unit.kilojoule_per_mole)
        beta = 1.0 / kt

        # 🔑 修复 Bug 1: 使用差分形式避免数值溢出
        # V_bias = -kT * ln( sum_k exp(-beta * (U'_k - f_k)) )
        # 提取 k=0 项: = -kT * [ -beta*(U'_0 - f_0) + ln( 1 + sum_{k>0} exp(-beta * ((U'_k - f_k) - (U'_0 - f_0))) ) ]
        # 简化后: = (U'_0 - f_0) - kT * ln( 1 + sum_{k>0} exp(-beta * (Delta_U_k - Delta_f_k)) )
        # 注意：OpenMM CustomCVForce 只能定义势能函数。
        # 我们定义: Energy = -kt * log( 1 + sum_{k>0} exp(...) )
        # 而 (U'_0 - f_0) 这一项实际上是一个依赖于坐标的项 (因为 U'_0 是 CV)。
        # 但是！IBS 的目标是平坦化分布。如果我们只关心相对权重，通常可以忽略全局偏移。
        # 然而，为了严格对应论文 Eq.18 并保证力的正确性，我们需要小心处理。
        
        # 更稳健的工程实现：
        # 令 X_k = cv_{k}_int + cv_{k}_rest - {prefix}_f_{k}
        # 目标势能: V = -kt * log( sum_k exp(-beta * X_k) )
        # 数值稳定写法: V = X_0 - kt * log( 1 + sum_{k>0} exp(-beta * (X_k - X_0)) )
        
        # 由于 OpenMM CustomCVForce 不支持直接在表达式中引用其他 CV 的复杂组合而不重复计算，
        # 我们采用标准的 Log-Sum-Exp 技巧。

        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12] 之前这里用 80.0*tanh(logit/80.0) 平滑
        # 饱和每个相对 logit，理由是"避免偏置力在边界处发生不可导突变"——但这个
        # 理由本身是错的：真正会引入不可导突变的是对 logit 做硬 min/max 截断
        # （clip 到固定范围），log-sum-exp 的 max-shift 技巧不是截断，是恒等变形。
        # 用真正的全局 max 作为平移基准时，V = M + log(sum_i exp(logit_i - M))
        # 对任意选取的 M（只要不溢出）都与 log(sum_i exp(logit_i)) 严格相等；
        # 对坐标求导时，dM/dx 项在展开后精确相消（标准 log-sum-exp 梯度=softmax
        # 的证明），即使 M 本身在 argmax 切换处不可导，组合表达式的力仍然精确
        # 等于 softmax 概率，处处连续。用旧的固定 k=0 基准（M=logit_0=0）只在
        # k=0 恰好是最大项时才安全，一旦某个 k 态在当前构型下比 k=0 更"划算"
        # （diff_k 很负、逻辑值很大正），单项 exp() 就可能溢出——tanh 饱和虽然
        # 防止了溢出，但把远端态实际贡献的力系统性压低了（80 kT 的饱和尺度只
        # 对应约 200 kJ/mol，vdW 生长/消失窗口在构象过渡瞬间的能量差很容易达到
        # 这个量级），这是一个真实的模型偏差，不是纯数值技巧——用真正的
        # max-pivot log-sum-exp 替换后，偏置力精确等于 MBAR/SGD/冻结验证共用的
        # 同一个 softmax 公式（见 update_weights/evaluate_frozen_batch_probability），
        # 三者不再是三套不同的数学模型。
        # EXP-025 G4 Layer-2: when residual_enabled, state k's parenthesized
        # term gains "+ (A_k)*(exp025_residual_basis-(offset))" before the
        # "- {prefix}_f_{k}" close. When NOT enabled, _state_expr(k) returns
        # EXACTLY the original "(cv_{k}_int + cv_{k}_rest - {prefix}_f_{k})"
        # string, byte-for-byte -- verified by
        # scripts/test_exp025_g4_ibsbiasforce_native_residual.py, not just
        # asserted here. Coefficients formatted with 17 significant digits
        # (round-trips a double exactly), matching the existing
        # OuterLambdaIBSBiasForce/OuterLambdaResidualBiasForce convention;
        # a state whose OWN coefficient is exactly 0.0 (e.g. a genuine
        # physical endpoint) omits the term entirely rather than adding a
        # numerically-inert "+0.0*(...)" -- also matching that convention.
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=30] 新增 {prefix}_s_residual：独立于
        # bias_scale 的残差耦合开关，默认 1.0（不显式调用 setParameter 时行为
        # 与 v29 逐字节相同）。原因见 2026-08-25 EXP-030 window_0 诊断：EM 阶段
        # bias_scale 仍是构造时默认值 1.0，candidate 在冷启动 f_k=0 时就已经用
        # 全强度残差力做无温控 L-BFGS 最小化，配对 baseline/candidate 密度探针
        # 显示两者在同一起点逐点一致，candidate 之后局部环境原子数雪崩、baseline
        # 全程稳定——但把 bias_scale 直接压到 0 会连 baseline 本来就在正常使用、
        # EM 时稳定贡献真实力的物理 softcore-state 混合项一起关掉，不是同一个
        # Hamiltonian。s_residual 只切残差项，EM 时物理软核混合力保持跟 baseline
        # 一样活跃。
        # [EXP-031 S1] `_rest_term(k)` is the ONLY difference between the two
        # Group-1 forms. In the legacy form it is exactly the literal
        # " + cv_{k}_rest", so both branches below reproduce the pre-EXP-031
        # expression strings BYTE-FOR-BYTE; in the omit form the identically-zero
        # placeholder simply disappears from the expression (and from the CV set),
        # which is an exact algebraic identity because cv_k_rest is a
        # CustomExternalForce("0") with zero particles -- see
        # IBS_BIAS_OMIT_ZERO_REST_CVS for the measured cost of keeping them.
        def _rest_term(k: int) -> str:
            return "" if self.omit_zero_rest_cvs else f" + cv_{k}_rest"

        def _state_expr(k: int) -> str:
            if self.residual_enabled and self.residual_state_coefficients[k] != 0.0:
                coeff = self.residual_state_coefficients[k]
                offset = self.residual_energy_offset_kj_mol
                return (
                    f"(cv_{k}_int{_rest_term(k)} + {prefix}_s_residual*({coeff:.17g})*"
                    f"(exp025_residual_basis - ({offset:.17g})) - {prefix}_f_{k})"
                )
            return f"(cv_{k}_int{_rest_term(k)} - {prefix}_f_{k})"

        logit_exprs = {}
        for k in range(1, n_states):
            # 相对 logit_k = -beta * (X_k - X_0)
            diff_expr = f"{_state_expr(k)} - {_state_expr(0)}"
            logit_exprs[k] = f"(-beta * ({diff_expr}))"

        # 平移基准 M = max(0.0, logit_1, ..., logit_{K-1})（0.0 对应 k=0 自身的 logit）
        pivot_candidates = ["0.0"] + [logit_exprs[k] for k in range(1, n_states)]
        pivot_expr = pivot_candidates[0]
        for cand in pivot_candidates[1:]:
            pivot_expr = f"max({pivot_expr}, {cand})"

        sum_terms = [f"exp(0.0 - ({pivot_expr}))"]
        sum_terms += [f"exp({logit_exprs[k]} - ({pivot_expr}))" for k in range(1, n_states)]
        sum_expr = " + ".join(sum_terms)

        # 最终能量表达式:
        # V_bias = X_0 - kt * (M + log(sum(exp(logit_i - M))))
        # 注意：第一项 (X_0...) 是坐标依赖的，必须包含在内以保证力的正确性。
        # _state_expr(0) already carries its own parens, matching exactly
        # what the original literal "(cv_0_int + cv_0_rest - {prefix}_f_0)"
        # substring looked like when residual_enabled is False.
        energy_expr = (
            f"{prefix}_bias_scale * ({_state_expr(0)} "
            f"- kt * (({pivot_expr}) + log(max(1e-300, {sum_expr}))))"
        )

        self.force = openmm.CustomCVForce(energy_expr)
        self.force.addGlobalParameter("kt", kt)
        self.force.addGlobalParameter("beta", beta)
        self.force.addGlobalParameter(f"{prefix}_bias_scale", 1.0)
        if self.residual_enabled:
            self.force.addGlobalParameter(f"{prefix}_s_residual", 1.0)
        for k in range(n_states):
            self.force.addGlobalParameter(f"{prefix}_f_{k}", 0.0)
        self.force.setForceGroup(1)
        if self.residual_enabled:
            # Registered here (inside __init__, not left to the caller) --
            # unlike cv_k_int/cv_k_rest (which the caller must still register
            # via addCollectiveVariable(), since only the caller has the
            # per-state softcore Forces to hand), this ONE shared CV is fully
            # determined by the constructor's own arguments, so there is no
            # reason to defer it and risk the exact "caller forgot to
            # register it" class of bug that G4 Layer-1 hit.
            self._cv_keeper.append(residual_basis_force)
            self._residual_basis_cv_index = self.force.addCollectiveVariable(
                "exp025_residual_basis", residual_basis_force
            )

    def addCollectiveVariable(self, name: str, cv_force: openmm.Force) -> int:
        if name == "exp025_residual_basis":
            raise ValueError(
                "IBSBiasForce: the 'exp025_residual_basis' collective variable name is reserved "
                "(registered automatically by __init__ when residual_basis_force is provided)"
            )
        # [EXP-031 S1] Fail closed: in the omit form this Force's own expression
        # does NOT reference any cv_k_rest, so registering one would (a) silently
        # cost a full inner-context force evaluation per step (21-25 us each,
        # measured; that is the entire point of the omit form) and (b) be reported
        # as `unexpected` by validate_wiring() below. Do not "tolerate" it.
        if self.omit_zero_rest_cvs and name.endswith("_rest"):
            raise ValueError(
                f"IBSBiasForce: refusing to register the collective variable {name!r} -- this Force was "
                "constructed with IBS_BIAS_OMIT_ZERO_REST_CVS=True, so its energy expression contains no "
                "cv_k_rest term at all. Registering one would pay a per-step inner-context force "
                "evaluation for an identically-zero contribution. Flip the module-level switch back to "
                "False if a caller genuinely needs the legacy placeholder slots."
            )
        self._cv_keeper.append(cv_force)
        # 索引必须是 self.force.addCollectiveVariable() 的**真实返回值**，不能事后按
        # 位置算（比如 2*k）——residual_enabled 时 __init__ 已经先注册了
        # exp025_residual_basis，会把后续所有索引整体偏移 1（第 3934 行那个调试分支
        # 用 idx_int=2*k 硬算就是这个坑，新代码不能重犯）。
        idx = self.force.addCollectiveVariable(name, cv_force)
        if name.endswith("_int"):
            self._int_cv_force_xmls.append(openmm.XmlSerializer.serialize(cv_force))
            # [性能修复：主 Context 一次读全部 CV] 记住这个 "_int" CV 在主
            # CustomCVForce 里的索引，供 IBSSampler._evaluate_interaction_energies_live()
            # 直接用 getCollectiveVariableValues() 一次性取全部值，不必再重建一个
            # 完全独立的 probe System/Context 逐个 force group 查询。
            self._int_cv_indices.append(idx)
        return idx

    def get_force(self) -> openmm.CustomCVForce:
        return self.force


    def get_residual_basis_energy(self, context) -> float:
        """Return the exact residual CV value used by the Group-1 score."""
        if not self.residual_enabled:
            return 0.0
        values = self.force.getCollectiveVariableValues(context)
        value = float(values[self._residual_basis_cv_index])
        if not math.isfinite(value):
            raise RuntimeError("non-finite residual basis energy")
        return value

    def get_sampling_state_coefficients(self) -> np.ndarray:
        values = self.residual_state_coefficients or [0.0] * self.n_states
        return np.asarray(values, dtype=np.float64)

    def setForceGroup(self, group_id: int):
        self.force.setForceGroup(group_id)

    def validate_wiring(self) -> None:
        """Fail closed BEFORE any Context is created if this Force's
        CustomCVForce is missing a collective variable or global parameter
        its own constructed expression actually references, or has a
        duplicate name registered. See
        outer_lambda_neural_basis.OuterLambdaResidualBiasForce.validate_wiring
        for why this exists (EXP-025 G4 Layer-1, 2026-08-13): a missing
        cv_k_int/cv_k_rest registration on that sibling class silently
        produced a wrong, finite energy with no exception at all. This
        class's cv_k_int/cv_k_rest are STILL the caller's responsibility
        (see addCollectiveVariable() above) -- only exp025_residual_basis is
        self-registered -- so callers must call this after registering all
        per-state CVs, before creating any Context.
        """
        expected_cv_names = {f"cv_{k}_int" for k in range(self.n_states)}
        # [EXP-031 S1] the omit form's expression references no cv_k_rest, so they
        # must be absent, not merely optional -- an extra one is `unexpected` and
        # rejected below (see addCollectiveVariable's matching fail-closed check).
        if not self.omit_zero_rest_cvs:
            expected_cv_names |= {f"cv_{k}_rest" for k in range(self.n_states)}
        if self.residual_enabled:
            expected_cv_names.add("exp025_residual_basis")

        actual_cv_names = [
            self.force.getCollectiveVariableName(i) for i in range(self.force.getNumCollectiveVariables())
        ]
        if len(actual_cv_names) != len(set(actual_cv_names)):
            duplicates = sorted({name for name in actual_cv_names if actual_cv_names.count(name) > 1})
            raise ValueError(f"IBSBiasForce.validate_wiring: duplicate collective variable name(s) {duplicates}")
        actual_cv_set = set(actual_cv_names)
        if actual_cv_set != expected_cv_names:
            missing = sorted(expected_cv_names - actual_cv_set)
            unexpected = sorted(actual_cv_set - expected_cv_names)
            raise ValueError(
                "IBSBiasForce.validate_wiring: collective variable set does not match this Force's own "
                f"contract -- missing={missing}, unexpected={unexpected}. Every cv_{{k}}_int/cv_{{k}}_rest "
                "for k in range(n_states) must be registered by the caller via addCollectiveVariable() before "
                "creating any Context."
            )

        expected_global_names = {"kt", "beta", f"{self.prefix}_bias_scale"}
        if self.residual_enabled:
            expected_global_names.add(f"{self.prefix}_s_residual")
        expected_global_names |= {f"{self.prefix}_f_{k}" for k in range(self.n_states)}
        actual_global_names = {
            self.force.getGlobalParameterName(i) for i in range(self.force.getNumGlobalParameters())
        }
        missing_globals = expected_global_names - actual_global_names
        if missing_globals:
            raise ValueError(f"IBSBiasForce.validate_wiring: missing required global parameter(s) {sorted(missing_globals)}")

    def set_bias_enabled(self, context: openmm.Context, enabled: bool):
        try:
            context.setParameter(f"{self.prefix}_bias_scale", 1.0 if enabled else 0.0)
        except Exception as e:
            print(f"  [WARN] 偏置开关设置失败: {e}")

    def update_parameters(self, context: openmm.Context, f_values: np.ndarray):
        for k in range(self.n_states):
            try:
                context.setParameter(f"{self.prefix}_f_{k}", float(f_values[k]))
            except Exception as e:
                print(f"  [WARN] 参数更新失败: {e}")


# The four-argument constructor is the stable public API. The runtime
# ``**residual_kwargs`` compatibility path above is intentionally kept out of
# introspection so existing API-contract checks and callers see that surface
# unchanged.
IBSBiasForce.__init__.__signature__ = inspect.Signature(
    parameters=[
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("n_states", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("temperature", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("prefix", inspect.Parameter.POSITIONAL_OR_KEYWORD, default="abfe"),
    ]
)

# 🔑 e_base/e_bias 的力组切分方式（哪些 Force Group 算"物理"、哪些算"纯采样偏置"）
# 每次改变时必须递增此版本号：resume/窗口产物复用逻辑靠这个字段判断一份旧的
# dual_window_*_base.npy/*_bias.npy/convergence.json 是否是在当前口径下算出来的，
# 不能靠"λ 值凑巧没变"就信任旧文件——决定物理量对不对的是"能量怎么分组"，不是 λ。
#   version 1: e_base=groups{0,2,3,4,5}, e_bias=groups{1}（Group4 λ-WCA 被误当 λ 无关）
#   version 2: e_base=groups{0,2,3,5},   e_bias=groups{1,4}（Group4 明确为纯采样偏置）
#   version 3: [2026-09-02] Group 4 **恒空**（WCA_SHIELD_RETIRED=True，防护壳退役
#              成死代码），因此 e_bias 实际只剩 {1}。`getState(groups={1,4})` 的写法
#              保持不变——空组贡献 0，且把常量翻回 False 复活防护壳时仍然正确。
#              升版的作用是让 v2 及更早的 dual_window_*_{base,bias}.npy /
#              convergence.json 一律失配：那些数据产自**含壳**的采样哈密顿量，
#              与本版本不是同一个物理系综，绝不能混用。
#              （system XML 指纹本来也会失效，这里是第二道显式判据。）
WCA_ACCOUNTING_VERSION = 3

# 🔑 [2026-09-02] e_base / e_bias 的力组集合**唯一定义处**。
#
# 加这两个常量的原因：这套切分原先以 `groups={0, 2, 3, 5}` / `groups={1, 4}` 的
# **字面量**散在 ibs_engine 的 3+3 处，外加 `outer_lambda_neural_basis.py`
# （神经/residual 臂的 `collect_energies`）里**又独立写了一份**，而那份完全不引用
# `WCA_ACCOUNTING_VERSION` —— 也就是说切分一改，版本号升了它也不会跟着失效。
# 力组语义是"哪些能量算物理、哪些算纯采样偏置"，一处不同步就是 Hamiltonian
# 记账不一致，而不只是效率问题。
#
# 规则：**改这两个集合必须同时递增 WCA_ACCOUNTING_VERSION**，并且所有取
# e_base/e_bias 的地方都必须用这两个常量，不准再写字面量。
IBS_E_BASE_FORCE_GROUPS = frozenset({0, 2, 3, 5})
IBS_E_BIAS_FORCE_GROUPS = frozenset({1, 4})

# 🔑 生产阶段在线 early-stop 判据协议版本。这个开关目前默认关闭
# （enable_early_stop=False），阈值默认值尚未用完整轨迹离线回放校准过——见
# run_all_windows 里对 early_stop_* 参数的说明。判据本身（哪些量、怎么组合、
# 连续通过要求）每次改变都必须递增此版本号：resume 时如果一份缓存是
# early_stop_triggered=True 提前停止产出的，只有 early_stop_protocol_version
# 与当前版本一致、且当前调用同样启用 early stop、且当前目标步数没有被调高，
# 才允许复用；任何一项不满足都视为缓存不可信，强制重新采样，避免"以后关闭
# early stop 或调高预算"时被一次短样本悄悄糊弄过去。
#   version 1: 初始版本——最小生产步数门槛 + 定期 local MBAR + 绝对 ESS/ESS
#       比例/去相关样本数/局部 ΔG 漂移/局部不确定度五项同时通过、连续
#       required_consecutive_passes 个独立 block 才停止；任一项失败清零连续
#       计数；硬上限仍是 n_steps_per_window（含 production_step_overrides）。
EARLY_STOP_PROTOCOL_VERSION = 1


def _early_stop_configs_match(cached_cfg: Optional[Dict], current_cfg: Dict) -> bool:
    """Compare a cached window's recorded early-stop threshold config against
    the current call's, tolerant only of float round-tripping through JSON
    (not of any actual difference in value).

    Matching ``early_stop_protocol_version`` alone is not enough: the
    protocol version only changes when the *decision logic itself* changes
    (which checks, how they combine). It says nothing about the specific
    threshold *values* used, which are ordinary function-default parameters
    a caller can change at any time without touching the protocol version --
    e.g. tightening ``max_uncertainty_kJ_mol`` from 5.0 to 1.0 between runs.
    A window that early-stopped and passed under the looser 5.0 threshold
    has no guarantee it would also pass under the stricter 1.0 one; reusing
    it silently would mean the "stricter" config never actually gets
    enforced against already-cached windows.
    """
    if not isinstance(cached_cfg, dict):
        return False
    if set(cached_cfg.keys()) != set(current_cfg.keys()):
        return False
    for key, current_value in current_cfg.items():
        cached_value = cached_cfg.get(key)
        if isinstance(current_value, float):
            if cached_value is None or not np.isclose(float(cached_value), current_value, rtol=1e-9, atol=1e-12):
                return False
        elif cached_value != current_value:
            return False
    return True


# 🔑 IBS 偏置预热/冻结协议版本。之前"续传时跳过 Warmup"和"生产阶段仍调用
# update_weights()"这两条合起来意味着：(a) 未真正收敛的偏置也会被当成已预热、
# 直接放行进生产；(b) f_k 在整个生产采样期间还在继续被 SGD 调整，同一个窗口
# 的样本实际上来自随时间变化的多个不同偏置势，而不是 MBAR 假设的单一固定
# 采样分布。现在改为：只有满足严格收敛判据（见 run_all_windows 里的收敛判据
# 说明）才允许进入生产，且进入生产后 f_k 冻结、不再更新。旧协议下产出的
# ibs_state_*.json（f_k 可能仍在漂移、且从未真正校验过是否收敛）不能被当成
# "已收敛"信任续传，必须靠这个版本号强制失效。
#   version 1: 续传即跳过 Warmup（不检查是否真收敛）；生产阶段仍调用 update_weights()。
#   version 2: 收敛判据升级为"逐态概率落在 (0.5/K, 2.0/K) 区间 + f_k 变化量收敛 + 连续
#              多次通过"，未达标直接 raise（不放行进生产）；生产阶段 f_k 冻结、只采样。
#              ⚠️ 有 bug：每 check_chunk=500 步检查一次，但 update_weights() 要攒够
#              10 帧才真正执行（约每 10 个 check_chunk 一次）——中间那几次检查看到
#              的是完全相同的旧 ema_mean_p/f_history，却仍被各自计成一次"通过"，
#              "连续 3 次"可能只是同一次统计结果被反复计数，不是三次独立证据。
#   version 3: 只有真正发生一次新的 update_weights() 调用（返回值非 None）才评估收敛，
#              没有新更新的迭代直接跳过，不计入连续通过也不清零。
#              ⚠️ 仍有缺口：load_ibs_state 只校验 n_states/prefix/协议版本，从未
#              校验 f_k 对应的实际 λ 值——λ 路径被自动加密/重新划分窗口后，某个
#              窗口凑巧还是同样的态数，旧 f_k 就会被当成有效热启动注入，但
#              f_k[k] 早就不对应同一个物理 λ 了，这是主动引入偏差，不是中性起点。
#   version 4: save/load_ibs_state 额外存取 lambdas_coul/lambdas_vdw，resume 时
#              与当前窗口的 λ 值逐一核对；n_states/prefix/协议版本/λ 值任何一项
#              不匹配，一律完全忽略旧状态（不再当热启动用），从 f_k=0 全新开始，
#              不再有"协议对不上但 f_k 仍可用"的中间态。
#   version 5: full-bias 收敛预算改按真实 update_weights() 次数计数；失败诊断保存
#              完整逐态 ema_mean_p 与相邻态 Delta-u 分布，并用结构化异常反馈路径规划器。
#   version 6: |Delta-f| 只保留为诊断量，不再把 SGD 步长本身误当收敛门；硬门改为
#              min(p_k)>0.5/K 且 1/sum(p_k^2)>0.8K，并要求连续真实更新通过。
#              大窗口失败只拆窗；三态及以下再用 fixed-H 双向 MBAR overlap 判断是否插点。
#   version 7: v6 的"连续通过"仍是假收敛——update_weights() 每次调用都用 f_old 算
#              概率、算完立刻把 f_new 写回 context，真正冻结进生产的那个 f_k 从未
#              被任何一次"通过"检验过。改为两阶段：learning（沿用 v6 判据找候选
#              收敛）→ freeze（冻结 f_k，绝不再 update_weights()，丢弃一段 burn-in
#              重新平衡）→ validate（用新增的只读 evaluate_frozen_batch_probability()
#              在完全固定的 Hamiltonian 下连续采若干新 batch 验证，失败则恢复
#              learning）。同时删除了"收敛后卸压"：旧代码在宣布 converged 后立刻
#              bias_scale 清零→最小化→5000 步无偏置运行→重新爬回 1.0→直接进
#              production，从未对这个被重新扰动过的构型复验；现在验证通过后只做
#              一次非破坏性安全力检查，不再改变 bias_scale/最小化/重新爬坡。
#   version 8: v7 的 frozen validation 判据本身又读了 evaluate_frozen_batch_probability()
#              的副作用 self.ema_mean_p，而不是该方法的返回值——真正生效的仍是
#              gamma=0.9 的滞后 EMA，一个塌缩 batch 可以被前几个正常 batch 的
#              EMA 记忆掩盖，等于把"用旧统计量冒充新证据"的 bug 原样搬进了刚写好
#              的验证阶段。改为直接用返回的原始 batch 概率判定，EMA 只作诊断。
#              同时修了 update_weights() 的同类问题：log_grad 之前也读
#              self.ema_mean_p（跨越不同 f_old 的滞后统计），现在改用当次真实
#              mean_p_batch；学习率衰减时标从 100 缩短到 30，并新增
#              self.eta_penalty（每次冻结验证失败恢复 learning 时减半），避免
#              "推过头->验证失败->反向推过头"的振荡；min_buffer_size 从 10 提到
#              20 降低单批噪声。新增：fixed-H 双向 overlap 全部通过时（证明 λ
#              网格本身没问题），直接用该 probe 已经解出的 BAR/MBAR 相邻态自由能
#              差校准 f_k（f_0=0, f_1=ΔF_01, f_2=ΔF_01+ΔF_12, ... 去均值），只给
#              一次冻结 burn-in + 独立验证的机会，不再回退 learning——比继续在一
#              个已经证明会振荡的 SGD 循环里盲搜更可靠。
#   version 9: fixed-H 双向 overlap 探针的触发边界从 K<=3 改为 K<=4，跟
#              abfe_preoptimizer.py 里 split_window_from_warmup_failure 的
#              min_states_before_split=5（v7 起）保持一致——K=4 的窗口不可能被
#              拆成两个各自 >=3 态、共享 1 态的孩子，之前 K<=3 的边界让 K=4 仍
#              走"直接拆窗"分支，产出过一个 K=2+K=3 的统计脆弱窗口；现在 K=4
#              和 K<=3 一样直接走这里的 fixed-H overlap 探针。诊断字段
#              feedback_action 的判定边界同步从 K>=4 改为 K>=5。
#   version 10: v8/v9 用来校准 f_k 的 delta_f_kJ_mol 来自"路径 overlap 探针"
#              （probe_bidirectional_overlap），这是一个真实的、已确认的 ensemble/
#              能量口径错配 bug，不是又一次假收敛：(1) 该探针的固定 Hamiltonian
#              动力学是 U_common + CV_k，_serialize_ibs_common_system 会连 Group 4
#              WCA 防护壳一起删掉——但生产实际采样的是
#              U_common + WCA_window(lambda_shield) + CV_k；WCA 只在"同一帧、
#              同一窗口内、不同态之间"的能量差里抵消，不会从采样动力学的构象
#              分布里消失，删掉它采的是另一个系综。(2) 该探针为了服务它自己的
#              "路径/λ 密度是否足够"这个不同问题，把 LRC 加进了喂给 MBAR 的能量
#              里；但生产 Group 1 偏置力本身从不含 LRC（LRC 只在
#              collect_energies() 里事后加进 target_energies 喂最终 MBAR/离线
#              分析，不进偏置力),用这个 delta_f 校准 f_k 等于把一个偏置力里根本
#              不存在的能量项校准了进去。(3) 门槛本身也不够：overlap>=0.03 只
#              证明两态"有最低程度的连接"，不证明 ΔF 已经精确到可以直接覆盖
#              f_k——真实案例 n_k_decorrelated=[39, 9]、overlap=0.105 时被直接
#              采信，去相关后只有 9 个独立样本的一侧显然撑不起这个精度要求。
#              (4) 校准冻结验证失败时，失败报告打印的 ema_mean_p[k] 仍是校准前
#              SGD 阶段的旧值，从未被这次校准验证真正观察到的最后一批概率覆盖，
#              诊断信息本身不可靠。
#              修复：新增独立函数 probe_bidirectional_overlap_for_bias_calibration
#              和 _serialize_ibs_common_plus_wca_system——校准探针的动力学换成
#              U_common + WCA_window(与生产同一个 lambda_shield=mean(该窗口
#              λ_vdw)) + CV_k，能量只用纯 softcore CV、绝不加 LRC，返回专用字段
#              delta_f_bias_kJ_mol（不是 delta_f_kJ_mol，两者不能再被混用）；
#              旧的 probe_bidirectional_overlap（不含 WCA、含 LRC）保留不变，
#              继续只服务"路径 overlap/λ 密度"这一个问题，两个探针的用途和实现
#              彻底分开。新增去相关样本数（默认 >=20）和 ΔF_bias 不确定度
#              （默认 <=1.0 kJ/mol）双重门槛，任一不满足就把 sample_steps 翻倍
#              重新采样（最多 3 次），仍不达标则放弃本次校准，不拿不够精确的
#              估计去覆盖 f_k。校准冻结验证失败分支现在会用 last_validation_batch_p
#              （本轮验证循环里真实观察到的最后一批概率）覆盖
#              ema_mean_p_values/min_ema_p/max_ema_p/coverage_ess，失败报告
#              不再打印校准前的旧值。
#   version 11: v10 的两个 fixed-H 探针按"边"（相邻态对）独立采样：K 态窗口的
#              K-1 条边各自建两个全新 Simulation、各自烧一遍 burn-in，共享同一个
#              态的相邻两条边把它重复采样了两次；bias 校准探针"不够精确就重试"
#              更是每次整段丢弃已采样本、从零重新烧 burn-in 再采样 20k/40k/80k。
#              修复：新增按"态"共享的可续采轨迹库
#              probe_adjacent_path_overlap_bank/probe_adjacent_bias_calibration_bank
#              （内部复用 _build_fixed_state_simulation/_extend_state_trajectory/
#              _analyze_adjacent_pair），K 态只建 K 个 Simulation、只烧一次
#              burn-in，同一态的轨迹同时喂给它左右两条边的 MBAR 计算——
#              _compute_bidirectional_overlap_from_u_kn 对每条边仍是完全自包含
#              的双态计算，不依赖另一条边怎么处理同一批帧，共享轨迹不改变任何
#              一条边自己的 u_kn/n_k 配对，只在假设性的"跨边合并不确定度"分析
#              里才需要留意相关性（目前代码库没有这么做）。旧的两个逐边探针
#              probe_bidirectional_overlap/probe_bidirectional_overlap_for_bias_calibration
#              原样保留，继续服务 abfe_pipeline.py::_probe_vdw_window_fixed_overlap
#              （生产后 ESS 修复用的是重新最小化过的独立起始构型，跟这里
#              warmup 阶段活体 context 的当前构型不是同一份，两个调用点的
#              checkpoint 不能混用，故本次只重构 run_all_windows 这一个调用点）。
#              随机种子从按边的固定 offset（0/1）改为按"全局态序号"
#              （global_state_index=start+local_index）生成，避免同一态在两个
#              不同边循环里的旧写法造成的重复初始化：
#              seed = BASE + 97*global_state_index（97 为质数步长，避免周期性
#              碰撞），四个独立 BASE 见 PATH_PROBE_INTEGRATOR_SEED_BASE 等常量。
#              轨迹库落盘于
#              checkpoints/probes/<stage_type>/window_<idx>/<path_probe|
#              bias_calibration_probe>/，manifest.json 记录系统/CV XML 哈希、
#              λ 数组、lambda_shield、温度/步长/摩擦系数、协议版本号等指纹；
#              任一字段不匹配则整个 window+probe_type 目录判定不可信、全部
#              重采（跟旧的 whole-window 缓存同一套"全对才信、错一项就整体
#              拒绝"哲学），指纹匹配后每个态各自独立判断能量/checkpoint 是否
#              完好。同一态若因 checkpoint 丢失被迫重开一个新 segment，两个
#              segment 分别独立去相关，绝不对跨越重启边界的拼接序列直接算
#              自相关（见 _decorrelate_per_segment）。
#              这是与本次轨迹库重构同时提交但逻辑独立的一处策略变更：早触发
#              条件默认由 IBS_EARLY_PROBE_TRIGGER_ENABLED 这个独立开关控制
#              （见该常量定义处），可以单独关闭以排查某次行为变化到底是
#              轨迹库重构还是早触发条件导致的。
#   version 12: 三处独立修复，均来自一次真实 GPU 生产日志（vdw 窗口 fixed-H
#               overlap 全通过、bias 校准 f_k 与物理相邻 ΔF 基本一致，但冻结
#               验证长期卡住，state 2 占据概率仅 ~0.0013）：
#               (1) IBSBiasForce 的偏置力表达式此前对每个相对 logit 做
#                   80.0*tanh(logit/80.0) 平滑饱和（理由是"避免 max/min 硬截断
#                   处不可导"），但这个理由本身错误——log-sum-exp 的 max-shift
#                   平移是恒等变形，其梯度经展开后精确等于 softmax，处处连续，
#                   不需要 tanh 近似；80 kT（约 200 kJ/mol）的饱和尺度在 vdW
#                   过渡构型的能量差量级上会被真实触发，系统性压低远端态应得
#                   的偏置力，可能正好阻止主轨迹跨越构象壁垒。现在改用真正的
#                   全局 max 作为 log-sum-exp 平移基准，偏置力精确等于
#                   update_weights()/evaluate_frozen_batch_probability() 用的
#                   同一个 softmax 公式，采样势/学习梯度/冻结验证概率三者
#                   不再是三套不同的数学模型。
#               (2) IBSSampler 新增 bias_status（"unconverged" /
#                   "calibrated_pending_validation" / "converged"）与
#                   frozen_f_k_pending，随 save_ibs_state/load_ibs_state 落盘/
#                   恢复。此前 MBAR 校准出的 f_k 一旦冻结验证在预算内未通过，
#                   就会和"从未校准过的普通未收敛"混为一谈地存为
#                   bias_converged=False；resume 时会被送回 [learning]，SGD
#                   重新调整这个其实已经正确的 f_k。现在校准成功但验证未完成
#                   会被落盘为 calibrated_pending_validation，resume 时直接
#                   带着冻结的 f_k 回到 freeze_burn_in/validating，不再回退
#                   SGD、也不再重跑 fixed-H overlap/校准探针。
#               (3) run_all_windows 冻结验证预算从单次固定的
#                   mbar_calibration_reserved_steps 改为可由调用方通过
#                   frozen_validation_step_overrides 按窗口累计延长（见
#                   abfe_pipeline.py::_run_stage_with_overlap_autorepair 里的
#                   calibration_pending_validation 分支，50k→150k→300k 阶梯）。
#                   【2026-07-16 更正】这条当时还写了"并在预算耗尽仍未通过时用
#                   OpenMM 原生 checkpoint（含积分器 RNG 状态）持久化主生产
#                   Context，下一轮真正续算而不是重新最小化、重新爬坡"——这句话
#                   描述的功能当时其实**从未真正实现**在主窗口路径里（只实现在
#                   探针轨迹库，见 _atomic_save_openmm_checkpoint 等），是一句
#                   写进注释但没有兑现的承诺，之前一直没被发现。真正的实现见
#                   下方"补丁 2"与 MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION。
#   补丁（未升版本号，仍是 12）：真实 GPU 生产日志发现 version 12 引入的
#     resumed_calibration_pending 续验分支有一个状态机 bug——续验冻结校准 f_k
#     时，只要 mode=="validating" 下单批验证没通过（`validation_pass_count==0`
#     分支，run_all_windows 内联循环），代码此前无条件 `mode = "learning"`，
#     没有检查 resumed_calibration_pending，等于让"从一开始就该永远停留在
#     freeze_burn_in/validating、绝不进 learning"（见上方 version 12 (2) 的
#     承诺）这句话形同虚设：验证一批失败就把 SGD 重新打开，续验预算
#     （frozen_validation_step_overrides 给的 50k/150k/300k 阶梯）几乎全被
#     无关的 SGD 学习烧掉，abfe_pipeline.py 的阶梯升级逻辑只读
#     calibration_pending_validation 这一个布尔值，分不清"真的验证了整段
#     预算仍失败"和"验证一批就被打回学习、剩余预算全在 learning 里空转"，
#     阶梯升到顶后会误报"冻结验证在 300000 步内失败"。修复：validating 分支
#     的失败处理现在按 resumed_calibration_pending 分两支——续验分支保持
#     mode="validating"、不重新 burn-in（Hamiltonian 全程未变，没有需要
#     burn-in 忘记的漂移历史）、不调用 apply_learning_rate_penalty()、不增加
#     learning_to_validation_cycles（该计数器语义是"退回 learning 的次数"，
#     续验分支必须保持为 0），改用独立的纯诊断计数器
#     frozen_validation_retry_count；全新窗口/skip_warmup_entirely 分支的原有
#     "验证失败退回 learning"行为不变。**刻意不升级协议版本号**：
#     bias_status/frozen_f_k_pending 字段的落盘格式和含义在修复前后完全没变，
#     旧 bug 也从未污染落盘的 frozen_f_k_pending（那份值来自循环开始前的
#     Python 快照，从未被误入 learning 的 SGD 触碰；唯一受影响的是
#     save_ibs_state 里 cosmetic 的顶层 "f_k" 字段，但 load_ibs_state 在
#     calibrated_pending_validation 分支会用 frozen_f_k_pending 覆盖注入，
#     这个漂移值从未被实际采信）——升版本号会让 load_ibs_state 的版本门控
#     整体作废现有的、已经过 fixed-H overlap+bias 校准探针证明过是对的
#     calibrated_pending_validation 缓存，逼对应窗口从 f_k=0 重新做一遍这些
#     真实物理探针（数十万步 GPU 时间），这正是这次修复要避免的浪费。
#   补丁 2（未升版本号，仍是 12）：上面 version 12 (3) 条声称已经实现的
#     "OpenMM 原生 checkpoint 持久化主生产 Context、下一轮真正续算"从未真正
#     兑现——resumed_calibration_pending 续验时，run_all_windows 每次仍然对
#     这个窗口重新最小化 + dt 测试步进 + Boresch 安全爬坡，即使窗口的构象/
#     速度/盒子在上一次尝试结束时早已充分平衡、冻结 f_k 完全没变（用户读代码
#     发现，已用只读 Explore agent 逐行核实：save_ibs_state/load_ibs_state 只
#     落盘 f_k/t/status，run_all_windows 窗口开头无条件重建 Context/最小化/
#     爬坡，跟 resume 无关）。现在把探针轨迹库已验证好的 native checkpoint
#     续算模式（_atomic_save_openmm_checkpoint/sim.loadCheckpoint，含积分器
#     RNG 状态）原样搬到主窗口：resumed_calibration_pending 续验 + 全新窗口
#     首次 MBAR 校准后验证，两条路径在每次评估完一个 batch 后都会覆盖式落盘
#     一份主窗口 checkpoint（新常量 MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION=1
#     管这份文件自己的指纹），下次 resume 若指纹匹配就直接 loadCheckpoint 跳过
#     最小化/dt测试步进/Boresch爬坡、且跳过 freeze_burn_in 直接从 validating
#     继续（Hamiltonian 全程未变，没有需要 burn-in 忘记的漂移历史）；checkpoint
#     缺失/不兼容才回退今天的完整重建流程（仍然不回退 SGD，见补丁 1）。窗口
#     真正收敛后清理这份 checkpoint。**同样刻意不升级
#     IBS_BIAS_PROTOCOL_VERSION**：这是纯增量能力，ibs_state_*.json 任何已有
#     字段的格式/含义都没变，用独立的 MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION
#     管新文件自身的指纹校验，没有历史缓存兼容性问题（这次上线前跑的窗口本来
#     就没有这份 checkpoint，第一次遇到自然回退到重建流程，无损）。
#   version 13：这次是真正需要升版本号的物理 bug——fixed-H overlap 探针/bias
#     校准探针（probe_bidirectional_overlap、probe_bidirectional_overlap_for_
#     bias_calibration、_build_fixed_state_simulation）新建 Context 后从未设置
#     lambda_boresch_scale，其 System 级默认值是 0.0（见
#     LambdaDependentBoreschForce.__init__），而主窗口生产/冻结验证早已爬坡到
#     1.0——探针证明的是关掉 Boresch 限制的系统，跟它要验证的生产 Hamiltonian
#     不是同一个。已修复三处探针的 Context 构建代码，显式对齐到 1.0。但
#     bias_status="calibrated_pending_validation"/frozen_f_k_pending 这些字段
#     本身的格式/含义没变，光改代码不会让 load_ibs_state 拒绝旧缓存——必须靠
#     这次版本号升级，让所有在这个 bug 修复之前就已经落盘的 frozen_f_k_pending
#     （物理上是用错误 Hamiltonian 证明过的，不能再信任）被版本门控拒绝，
#     强制从 f_k=0 重新做一遍这些真实物理探针，而不是继续沿用它们。
#   version 14：non_mutating_v1 把 sampling_repair_policy 加入 ibs_state 和
#     convergence 身份指纹。这个字段决定 f_k 是否可能被旧的 fixed-H/MBAR
#     变异修复路径就地改写，因而是采样协议的一部分，不能在不升版本号的情况下
#     要求所有既有 v13 state 都已经带它。旧实现漏升版本且 load_ibs_state 又在
#     协议版本门之前检查该字段，导致本应按正常版本失配路径安全失效的 v13 state
#     （sampling_repair_policy=None）被提前劫持成
#     ExistingEnsembleRequiresRescueAudit，整条 resume 流程无条件硬停止。v14 同时
#     修正门控顺序：先确认 n_states/prefix/协议版本/λ 身份，只有一份确实声称属于
#     当前 v14 ensemble 的 state 才检查 repair policy。这样 v13 数据绝不注入
#     Context、会从 f_k=0 重建；而缺字段/legacy policy 的 v14 数据仍然 fail-closed。
#   version 15：收敛门改为直接检查论文中的 Log-Sum-Exp 自洽方程。
#     对每个冻结 batch 计算 r_k=log(K*<p_k>)，只有 max|r_k| 不超过显式容差
#     才算一次通过；learning 候选判定也读取当前 update 的原始 batch，而不再用
#     跨 Hamiltonian 的 EMA。旧的 min(p)>0.5/K + coverage_ESS>0.8K 只是宽松覆盖
#     代理，不能证明 LSE fixed point 已解出。v14 的 bias_converged 状态没有通过
#     这条方程门，必须失效重验。
#   version 16：修正“判据是 LSE、更新却仍是衰减 SGD”的不一致。update_weights
#     现在直接应用 f_k<-f_k-kT*log(K*<p_k>) 的完整固定点步，不再乘 eta_sgd；
#     bias_scale<1 的 ramp 只做动力学爬坡，不采集正式方程样本、不修改 f_k；
#     non_mutating_v1 不会进入 legacy MBAR 校准，因此不再从 500k 预算中空留
#     50k，max_bias_updates=50 时能真正执行满 50 次而不是 45 次。v15 的 f_k
#     来自不同更新方程且可能在 45/50 提前终止，必须失效重学。
#   version 17：v16 错把单个 20-frame batch 的 <p_k> 当成论文 eq. 13 的
#     time-averaged Q_hat_k，并对这份高方差估计施加完整固定点步，真实运行表现为
#     左右端占据交替饱和（max LSE residual 最高 33.7），而不是收敛。现在每个
#     iteration 用采样时的 n_k=exp(beta*f_k) 做重要性校正：Q_k ∝ <p_k>/n_k，
#     在 log-space 去掉各 iteration 公共归一化后跨迭代等权累计 Q_hat_k，再按
#     n_k ∝ 1/Q_hat_k 更新。learning 候选门读取“更新前的 n_old*Q_hat_TA”平衡
#     残差；冻结后仍用全新 raw batch 直接检查 log(K*<p_k>)，不把更新后恒等于零
#     的代数结果冒充独立收敛证据。TA 累加器随 ibs_state 落盘，v16 state 因更新
#     方程和状态语义不同必须失效重学。
#   version 18：v17 虽然开始累计 Q_hat，却把每个 iteration 的 <p_k>/n_k
#     分别归一化后再平均，漏掉 normalized p_IBS^(t) 的配分函数
#     Z_t=sum_j n_j Q_j。真实冻结验证因此连续 9 次显示候选 f_k 完全错误，而
#     learning 内部的 n*Qhat_TA 残差仍因自归一化惯性虚假接近 0。现在用上一轮
#     TA Q_hat 计算 Zhat_t，再累计 Zhat_t*<p_k>/n_k；同时冻结验证在同一 f_k 下
#     累积多个 batch，不再因单个 20-frame batch 未通过就立刻退回 learning。
#     v17 TA 累加器的尺度/含义不同，必须由版本门控失效。
#   version 19：v17/v18 的 TA（时间平均）估计器只用论文 eq. 11-13，每轮更新完
#     就把这批原始能量丢掉（self.energy_buffer=[]），没有机制把冷启动早期
#     （f_k=0，state 2/3/4 采样天然稀疏）产生的偏置 mean_p_batch 后续"稀释"
#     掉——TI 交叉验证证实 window 0 观测到的 state0↔state5 ~300+ kJ/mol 落差
#     （真实 TI 只有 ~16.7 kJ/mol）就是这个数量级偏差的病理，不是真实物理量。
#     论文 eq.15 TMBAR（本仓库已经实现、已经在 GlobalMBARAnalyzer.
#     solve_stage_integrated 里跑生产/最终分析）才是"聚合来自一系列时变 IBS
#     分布样本"的正确工具。现在 update_weights()/冻结验证失败后的 feedback
#     两个调用点都不再丢弃 minibatch：每批 (u_kn, bias_energies, base_energies)
#     打包成 lambda_indices 固定为本窗口全部态索引的一条 entry，追加进随
#     ibs_state 落盘的持久列表 self.tmbar_history；f_k 由
#     solve_stage_integrated(self.tmbar_history, kt) 的 f_k（mean-centered）
#     给出，learning 候选门改用它的四项联合 converged（ESS ratio+绝对 ESS+
#     去相关样本数+端点不确定度），比旧的单一 LSE 残差更严格。彻底删除
#     ibs_lse_time_averaged_update 及 ta_relative_q_sum/ta_iteration_count——
#     两者语义/尺度都跟 tmbar_history 不同，必须由版本门控失效，不能被静默
#     当成空的 tmbar_history 复用。
#   version 20：vanishing 窗口 0 在 v19 之后仍反复 IBSWarmupConvergenceError
#     （6/4/3 态分组、pilot 自适应加密、真实 Δf 均匀切分五次真实 GPU 尝试全部
#     失败，占据始终卡在 state0 ~96-99%，`min_absolute_ess`≈1.0）——根因确认
#     不在 λ 网格，而在于每个全新（非 resume）窗口的 f_k 从 OpenMM 力常量默认值
#     0.0 冷启动（`ibs_engine.py:2473`），SGD/TMBAR 只能从每批 ~20 帧里的统计
#     证据自举出需要多大的偏置——但如果构型跨越本来就难采，早期批次几乎看不到
#     欠采样态的证据，学不出足够大的偏置，也就永远看不到更多证据（自举困境，
#     与 λ 摆在哪里无关）。修复：(1) 新增
#     `abfe_preoptimizer.estimate_f_k_from_pilot_ti`，对 Stage 2 pilot 早就测过
#     的真实平均梯度 `mean_dU_dlambda_kJ_mol`（不是这份 changelog 之前一直用的
#     方差代理 `metric_g`）做热力学积分得到 F(λ)，插值到窗口自己的 λ 值上，
#     mean-centered 后作为该窗口第一次 learning 的 f_k 起点，而不是从 0 开始
#     摸索——这份数据在窗口第一次真正采样之前就已经存在（pilot 阶段测的），
#     不需要先失败一次才能用。`IBSWindowManagerDualLambda` 新增
#     `pilot_lambdas`/`pilot_mean_dU_dlambda`（默认 None，向后兼容），
#     `run_all_windows` 只在 `not is_resumed_ibs`（真正全新学习、没有可复用
#     resume 缓存）时注入，不影响 resume 语义。(2) 候选门的
#     `candidate_min_absolute_ess` 从 2.5 降到 1.0：真实失败诊断显示四项候选
#     判据里 min_overlap/去相关样本数/端点不确定度三项都已经过关，只有
#     min_absolute_ess≈1.0（有效样本数的数学下限，只要还有非零权重样本就不可能
#     低于 1.0，本身不是这个区间里的质量信号）卡在 2.5 这个门槛——2.5 在这个场景
#     下意外地不可满足，不是有意设的质量要求，调到数学下限 1.0 不影响其余三项
#     判据，也完全不碰独立、未变的 `final_*`/冻结验证严格门槛。
#   version 21：v20 warm-start 上线后第一次真实 GPU 测试，占据反而从 96.8% 恶化
#     到 99.1%（state1 min_absolute_ess 降到 8.6e-6）——排查发现 v19 起
#     `_solve_tmbar_and_recenter` 一直直接使用
#     `solve_stage_integrated`（`df_matrix[0, 1:]`，pymbar 原始
#     `f_i=-ln(Z_i)` 相对采样态的约定）的原始输出作为 f_k，未反号。但
#     `IBSBiasForce`（本文件 2397 行起）的能量表达式与 `update_weights`
#     的 `logits = beta*(f_old - u_mk)`（约 3364-3367 行）严格对应
#     `n_k=exp(beta*f_k)`：给某态更高的 f_k 直接推高它在混合分布里的权重。
#     `solve_stage_integrated` 原始约定对越贴近当前采样分布（即当前过量
#     代表的态）给出越高（越不负）的值——用当前失败窗口真实数据验证：
#     99.1% 占据的 state0 被判给全组最高（最不负）的 f_k，直接喂给
#     context.setParameter 等于每次在线更新都在进一步强化已经占主导的
#     态，而不是压低它。这个符号错误从 v19 引入 TMBAR 起就存在，影响的是
#     这条在线学习路径覆盖的所有窗口，不只是 window 0。修复：
#     `_solve_tmbar_and_recenter` 现在对 `solve_stage_integrated` 的原始
#     f_k 取负号后再 mean-center。同时暂时禁用 v20 引入的 pilot TI
#     热启动注入（`run_all_windows` 里 `estimate_f_k_from_pilot_ti` 调用
#     点，见该函数旁注释）——pilot 的 `mean_dU_dlambda_kJ_mol` 是每个 λ
#     独立短程弛豫后测的系综平均梯度，和 update_weights 实际用的 u_mk
#     （在当前占主导构型上跨态求值）是概念不同的量，其本身是否同样反号
#     未经独立验证，先只验证 TMBAR 符号修复这一个变量，避免两处不确定
#     叠加。（历史记录：这项 v21 符号判断后来由 v27 用
#     exp(beta*f_k)Z_k=exp[beta*(f_k-F_k)] 恒等式及 solve_stage_integrated 的
#     明确返回单位证实为错误；v27 已撤销该反号。）
#   version 22：独立用平衡条件 `f_k - U_k ≈ const` 核对 v20 的 pilot-TI
#     warm-start，确认它和 v21 修复的 TMBAR 调用点存在同一个符号错误：原始
#     pilot TI 曲线也会把最高 f_k 给已经占主导的 state0，从而通过
#     `n_k=exp(beta*f_k)` 进一步放大其权重。现在
#     `estimate_f_k_from_pilot_ti` 对原始 TI 插值结果取负号后再 mean-center，
#     返回可直接注入的 bias-parameter convention；同时重新启用
#     `run_all_windows` 的 pilot-TI 注入。此前实测的 v20 window 0 种子
#     `[+48.2, +35.3, +20.0, +4.3, -15.2, -35.9, -56.7]` 因而变为
#     `[-48.2, -35.3, -20.0, -4.3, +15.2, +35.9, +56.7]`，与 v21 的
#     修复方向一致：已经占主导的 state0 得到全组最低 f_k。v21 的 TMBAR
#     修复本身不变。（历史记录：v27 证实这里把“观测占据的负反馈方向”误套到
#     “物理自由能种子方向”；pilot/TMBAR 两处反号均已在 v27 撤销。）
#   version 23：v22 首次真实 GPU 运行把占据从 state0≈99% 推到了 state6≈99.5%，
#     证明 pilot-TI 热启动成功让体系离开旧主导端，但在线更新仍把另一端锁死。
#     失败状态的 f_k 已发散到 [-1197.6, ..., +296.1] kJ/mol；根因不是“反号
#     还不够”，而是 update_weights 把每次尚未收敛（min_absolute_ess≈1）的
#     TMBAR 绝对向量整组覆盖进 Context。mean-center 只消除规范零点，并不会
#     缩小跨度或平均权重。现在 TMBAR 继续累计全部时变分布样本，但只承担候选
#     覆盖质量门/诊断；实际 f_k 学习改为对每批真实占据执行全态同步负反馈：
#     delta_f_k=-eta*kT*log(K*<p_k>)，log 残差和每轮最大步长都显式受限，
#     再 mean-center。过强态必降、过弱态必升，不再把任何正号/反号的 TMBAR
#     绝对解直接覆盖到 f_k。冻结验证失败的累计占据也走同一条受限更新，并降低
#     eta_penalty。pilot-TI 仍只负责提供非零起点；v22 状态必须失效，不能续用
#     已经发散的 f_k/tmbar_history。
#   version 24：v23 上线后 window 0 反复 5 次以上完整 learning->freeze_burn_in->
#     validating attempt 全部失败，从未进入生产。真实失败快照
#     （dual_window_0_vdw_warmup_failure.json）证实：K=7 时 state6 占据 p 高达
#     0.6-0.68（目标 1/7≈0.143），需要的 delta_f_6 约 3.6 kJ/mol，整条 f_k
#     向量量级在 10-50 kJ/mol，而 `_bounded_log_occupancy_update` 每步只挪了
#     ~0.1-0.15 kJ/mol（远低于当时 max_step_kT=2.0 换算出的 4.989 kJ/mol
#     上限）。根因是两处学习率抑制叠加过猛：(1) `apply_learning_rate_penalty`
#     每次冻结验证失败就把 `eta_penalty` 砍半、floor=0.05——5 次真实失败后
#     0.5**5=0.03125 已跌破 floor，之后每次 resume 都从磁盘原样加载这个
#     触底值（`save_ibs_state`/`load_ibs_state` 持久化 `eta_penalty`），
#     没有任何机制能让它回升，等于把所有未来 learning 尝试永久限制在 1/20
#     步长；(2) `update_weights` 里 `eta = eta_penalty/(1+update_index/100)`
#     的 /100 分母在验证预算只有 50000 步（≈100 frames）内进一步快速衰减
#     可用步长。三者叠加：100 frames 预算、0.1 kJ/mol 步长、0.25 的 LSE
#     容差，数学上不可能在预算内收敛——每次"3 步一冻结"验证必败，验证失败
#     又再砍一次学习率，形成预热步数远超生产步数（本窗口生产仅
#     50000 步/window）却永远进不了生产的恶性循环。修复：
#     `apply_learning_rate_penalty` 的 factor 从 0.5 松到 0.85、floor 从
#     0.05 提到 0.25（`_load_ibs_state` 校验区间同步改为 [0.25, 1.0]）；
#     `update_weights` 的衰减分母从 /100 放缓到 /500；
#     `_bounded_log_occupancy_update` 的 `max_step_kT` 默认值从 2.0 提到
#     5.0（单步硬上限约 12.5 kJ/mol），让单步物理上能够覆盖真实需要的
#     跨态修正幅度。不改变收敛判据本身（LSE 自洽 `max|log(K*p_k)|<=0.25`
#     仍是论文要求、未放松）——只是让 SGD 真正有能力在预算内到达那个判据，
#     而不是被人为砍到几乎为零的步长锁死。旧协议下已经触底/发散的
#     eta_penalty/f_k 状态必须失效，不能续用。
#   version 25：v24 上线后真实 GPU 测试暴露一个新问题——delta_f 确实跳到了
#     2-5 kJ/mol（符合预期），但 dominant_k 每次更新都在不同态之间跳变
#     （state4→state3→state6→state2→...），三次完整冻结验证 attempt 全部
#     失败，占据从未稳定下来。根因：`_bounded_log_occupancy_update` 同时对
#     全部 K 个态按各自当前残差施加 delta_f（同时压低占主导的态、抬高多个
#     欠采样的态），这在 softmax 耦合系统里会把相对 log-odds 的移动幅度
#     放大到远超公式隐含的单态线性响应假设——v24 去掉了 eta_penalty 快速
#     衰减到底这个（虽然调错了方向，但确实存在）的阻尼，而没有补上任何替代
#     阻尼，暴露出这个此前一直被过度阻尼掩盖的过冲问题。修复：新增独立于
#     eta_penalty（失败计数机制）和 update_index 衰减（一次 learning
#     attempt 通常只有 12-18 次更新，/500 分母基本不起作用）的
#     `IBS_UPDATE_RELAXATION_FACTOR=0.35` 阻尼系数，直接乘进 eta——该数值
#     未经解析推导，是经验性阻尼系数，若后续真实 GPU 测试仍观察到
#     dominant_k 反复跳变（下面新增的 watchdog 会打印警告），应进一步调低
#     （如 0.15-0.2）；若收敛明显变慢，可适当调高。同时新增 dominant_k
#     跳变 watchdog（纯诊断，不影响控制流）。
#     同一次改动里一并处理三个之前讨论过的独立问题（均已获用户确认）：
#     (a) `lse_log_residual_tolerance` 从 0.25 放宽到 0.5，并新增"有界次数
#     后接受目前最好结果"的兜底（不再无限重试）——两份独立研究证实：论文
#     本身用固定步数预算而非强收敛判据放行生产（integrated-boltzmann-
#     sampling.md 336/348 行）；占据均衡只影响重加权效率/方差，不影响
#     `solve_stage_integrated`（单参考重要性 MBAR）的无偏性；真正的正确性
#     门槛是完全独立于这个预冻结判据的生产后 `_assert_stage_result_sane`
#     （`min_overlap`/绝对 ESS/去相关样本数/端点不确定度，abfe_pipeline.py
#     2641-2644/3340-3436），任何一次"差不多就放行"的 f_k 若真的采样不足，
#     会在那里 fail-closed，而不是静默产出错误结果。因此放宽这里的判据只
#     影响效率，不影响正确性。
#     (b) 实现此前草拟但未落地的 PROPOSAL_frozen_validation_fallback.md：
#     learning 阶段候选连续通过次数从未凑满、纯粹因为 `max_bias_updates`
#     耗尽而 break 时，现在给当前 f_k 一次真正的冻结验证机会（仅一次，
#     guard 防止重复），而不是直接判定 f_not_converged——burn-in/validation
#     预算本来就已经预留，不用白不用。
#     (c) 当时曾把冻结验证阶段样本结转进生产数据集；该调度优化后来撤销：
#     预热/验证与生产必须是两个严格隔离的数据阶段，验证样本一律不计入生产。
#   version 26：v25 上线后第一次真实 GPU 全流程测试——5 个 vdw 窗口全部
#     真正跑完了预热+生产，但 Stage 2 最终 `_assert_stage_result_sane`
#     报告 `min_overlap=0.0047`（阈值 0.05）而失败。查真实落盘诊断：5 个
#     窗口全部是 `learning_to_validation_cycles==2`（v25 默认的 cap）就被
#     best-effort 接受，但接受时的 `best_effort_residual` 高达 13~123
#     （容差 0.5 的 26~246 倍），对应真实占据概率低到 1e-10~1e-54 的态——
#     不是"差不多"，是这些态在生产阶段几乎完全没被采样。根因：v25 只加了
#     "重试次数上限"，没加"接受结果本身是否合理"的下限检查，等于把
#     "见好就收"实现成了"不管多差、够了次数就收"。同时 cap=2（每次 cycle
#     只给 learning ~12-18 次更新的机会）本来就远不足以让这类大跨度
#     （50+ kJ/mol）不均衡收敛。修复：cap 从 2 提到 8（f_k/eta_penalty 跨
#     cycle 累积，更多 cycle 才有机会让有界负反馈逐步纠正过来）；新增
#     `best_effort_max_residual_multiple=4.0`——只有达到 cap 时最优 attempt
#     的残差仍在 `tolerance` 的 4 倍以内，才真正采用 best-effort 路径；
#     否则视为"目前最好的也远未达标"，不静默放行，回退到 v25 之前的原有
#     行为（继续 learning，预算耗尽后按 non_mutating_v1 fail-closed 抛
#     IBSWarmupConvergenceError，交人工/rescue 审计）——绝不通过放宽下游
#     `_assert_stage_result_sane` 的 min_overlap 阈值来掩盖这类真实的
#     欠收敛，那样会让统计上不可靠的态悄悄进入最终 ΔG。
#   version 27：修复 pilot-TI 热启动和 TMBAR 候选向量的符号。IBS 势
#     V=-kT*log sum_k exp[-beta*(U_k-f_k)] 下，态 k 的积分权重正比于
#     exp(beta*f_k) Z_k = exp[beta*(f_k-F_k)]，所以平坦权重要求
#     f_k=F_k+constant。旧 estimate_f_k_from_pilot_ti 对物理 TI 曲线额外取负，
#     会把本来低自由能/高占据的态进一步抬高；真实 window 1 因此从跨度
#     52.61 kJ/mol 的完全反向种子出发，26 次有界负反馈主要耗在撤销错误热启动。
#     v27 删除该反号；同时删除 `_solve_tmbar_and_recenter` 对
#     `solve_stage_integrated` 返回的物理 F_k 再次施加的反号。后者从 v23 起只作为
#     未安装进 Context 的诊断候选返回，本次失败不是它造成的，但保留会给未来调用
#     埋下同一符号错误。在线占据负反馈 delta_f=-eta*kT*log(K*p_k) 保持不变。
#     升版使所有由旧反号种子生成的 IBS 状态/生产缓存失效，避免静默续用。
#   warmup-control follow-up（不改变生产 Hamiltonian/cache identity，故协议仍为
#   version 27）：修复多次冻结验证与总预算不相容导致的“合理结果来不及放行”。
#     v26 允许最多 8 次完整冻结 attempt 后接受 sane best-effort，但总步数只额外
#     预留了 1 次 burn-in+validation；真实 v27 run 在 4 次完整 attempt 后，第 5
#     次只采到 20 帧便于 mode=validating 撞总帽，永远到不了 8 次分支。现在总帽
#     耗尽时也可复用同一 sane-bound，但只接受此前完整采满 attempt 预算的最佳
#     fixed-f 结果，并额外要求累计 TMBAR converged；当前被截断的半次验证绝不参与
#     选择。若恢复的是较早 attempt 的 f_k，先在该 f_k 下重新 burn-in、丢弃这段
#     轨迹，再进 production。生产后独立 overlap/ESS/去相关样本/不确定度硬门不变。
#     同时修正 dominant-flip watchdog：接近平坦时 argmax 随噪声换人不是过冲，
#     只有 dominant 概率至少达到目标的 1.5 倍时才累计/报警（仍纯诊断）。
#   version 29：把 warmup 的 f_k 收敛判据整体换成"局部滑窗 MBAR loose-gate"，
#     彻底移除旧的冻结验证机制（累计 <p_k> 的 LSE 自洽门 max|log(K·<p_k>)|≤tol、
#     required_consecutive_bias_updates 连续通过、50k→150k→300k 冻结验证阶梯、
#     "见好就收"best-effort、warmup ESS/coverage 四联门）。新判据：冻结 f_k、短
#     burn-in 后累计最近 IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES 批固定-f_k minibatch，
#     跑一次单参考 local MBAR（_solve_single_window_local_mbar，不吃全历史时变
#     TMBAR），只设一个 loose gate：max_k|Δf_{k,k+1}−ΔF^MBAR_{k,k+1}|（相邻差、
#     gauge 无关）< IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL(10 kJ/mol≈4 kT)
#     即冻结进生产；≥阈值退回 learning 再做一轮现有 update_weights；MBAR 不可解/
#     NaN 也退回继续更新。跑满 warmup 步数预算仍未通过则接受当前 f_k 进生产
#     (best-effort)。动机：避开时变 TMBAR 判据不稳/自身 BAR 不收敛导致的"合理
#     f_k 永远卡在验证门外"，把真正的自由能/ESS/overlap/误差全部交给生产后独立的
#     _assert_stage_result_sane + 最终 MBAR。这是纯 warmup 停止判据/诊断变更，不动
#     production Hamiltonian、f_k 符号约定、采样方式；且新 loose gate 严格弱于旧
#     LSE 门（旧门通过的近均匀占据 f_k 必然满足新门），故按 v28 同样理由保持缓存
#     兼容——已完成/已收敛的 v27/v28 f_k 不判废、可直接续用。
#     v29 同批还修了 warmup 更新控制器的塌缩正反馈（"低重叠 TMBAR 给出错误大步 →
#     占据塌缩 → TMBAR 更不可靠"）：(a) update_weights 只有累计 TMBAR 解质量同时
#     可信（min_overlap≥0.05、min_absolute_ess≥10、去相关≥10、端点不确定度≤5
#     kJ/mol、当前 batch coverage_ESS≥0.8K，见 IBS_TMBAR_TRUST_*）时才应用
#     _damped_tmbar_absolute_update 的绝对候选；否则退回 _bounded_log_occupancy_
#     update 先恢复覆盖。(b) 任何一次应用到 Context 的更新，相邻步长
#     max_k(Δf)-min_k(Δf) 硬 cap 到 IBS_MAX_APPLIED_PAIRWISE_STEP_KT=2 kT（在
#     update_weights 里、两个控制器之后统一施加，不进 _damped_* 本体），damping
#     0.20→0.10。(c) learning minibatch 20→40 帧（配 5 批滑窗=200 冻结帧）。
#     (d) local-MBAR loose gate 增加可信度门（min_ess_ratio<0.05 或绝对 ESS<10 →
#     insufficient_overlap，不把零重叠外推的巨大 ΔF 当门残差）。这些同样只改
#     warmup 学习/停止动力学与诊断，不动 production Hamiltonian/f_k 约定/采样，
#     故仍保持缓存兼容。
# v30（2026-08-25）：新增 {prefix}_s_residual，最小化阶段对 residual_enabled
# 的 arm 临时关闭残差耦合（EM 时 bias_scale 仍是构造默认值 1.0，candidate 在
# 冷启动 f_k=0 下会用全强度残差力做无温控 L-BFGS 最小化；配对 baseline/candidate
# 密度探针显示同一起点逐点一致、candidate 之后局部环境原子数雪崩撞上
# max_environment_atoms 硬上限、baseline 全程稳定，见 EXP-030 window_0 诊断），
# 最小化后（四条 is_resumed_ibs 分支的任意一条、或没走这四条分支时紧跟着的
# "[偏置预热]"爬坡块）统一恢复到 1.0。这是真实的 production Hamiltonian 变化
# （EM 阶段的有效势能变了，不是纯 warmup 停止判据/诊断），不能跟 v27-29 一样
# 保持缓存兼容——residual_enabled 的窗口如果在 v30 之前已经跑过 EM，那次 EM
# 是在全强度残差力下做的，跟 v30 的物理过程不是同一件事，旧 f_k 不能被当成
# v30 的有效热启动/冻结验证结果直接续用。baseline（residual_enabled=False）
# 不受影响：s_residual 只在 residual_enabled 时才注册和被设置。
# v31（2026-08-26）：修复 f_k 在线学习目标与实际部署目标不一致的 bug——
# collect_energies() 里喂给 self.energy_buffer（进而 update_weights()/
# _solve_tmbar_and_recenter() 的 u_mk）的量此前一直是纯物理 softcore_energies，
# 完全不含残差项；但实际驱动采样动力学的 Group-1 偏置力公式是
# X_k = U_k + s_residual·A_k·B_φ − f_k（v30 引入 s_residual 时改的
# _state_expr(k)）。residual_enabled 的窗口因此在拿一份不含残差的训练信号去
# 学习如何拉平一个含残差的分布，学出来的 f_k 系统性学不对——EXP-030 生产数据
# 三个独立 repeat 的 candidate 臂占据分布系统性偏斜（跟 baseline 的近似均匀
# 形成鲜明对比）证实了这一点。修复后改用已经算出来但此前只存进
# self.sampling_state_energy_history、从未接入训练链路的 sampling_state_energies
# （含残差，full-strength，不随 bias_scale/s_residual 爬坡缩放——这是有意的，
# 物理项那边训练目标本来就不跟着 bias_scale 爬坡缩放）。这是真实的训练目标
# 变化：v30 之前用 residual_enabled 学出来的所有 f_k，都是在错误目标下学的，
# 不能被当成 v31 的有效热启动/冻结验证结果直接续用。baseline
# （residual_enabled=False）不受影响：sampling_state_energies 在这种情况下
# 就是 softcore_energies 的原样拷贝，这次改动对 baseline 逐字节不变，但为了
# 不悄悄放过 candidate 窗口，版本号检查本身不区分 arm，直接把兼容集合收窄成
# 只有 v31 自己，代价只是 baseline 也会多冷启动一次（跟 v30 断点时的处理方式
# 一致）。
# v32（2026-08-31）：修复逐帧 e_offset 从 energy_buffer 泄漏进 tmbar_history 的
# u_kn（0831issue.md P1）。`_append_tmbar_batch_from_buffer()` 此前把已减逐帧
# `e_offset` 的 `self.energy_buffer` 当 u_kn 存盘，却配未偏移的 bias/base，
# 于是增广矩阵里目标行含 −c_n、采样行不含，这个逐帧平移在 MBAR 里不抵消，
# 等价于人为注入一个共模因子。4W53 实测 sd(c_n)/kT = [3.10, 1.98, 1.76, 0.84,
# 0.33, 0.13]（window 0→5），耦合端比真实防护壳共模（0.95~2.40 kT）还大；它把
# raw 单参考 ESS 压到个位数，使 IBS_TMBAR_TRUST_MIN_ABSOLUTE_ESS 几乎永不满足
# （可信绝对 TMBAR 控制器失效、永远退回 bounded occupancy），并让 v29 loose
# gate 的 ΔF^MBAR 偏差与真实物理脱钩。修复后 u_kn 取未偏移的
# sampling_state_energies。
# 这与 v31 同类：**在线学习/冻结判定的输入变了**，旧 f_k 不能当作 v32 的有效
# 热启动或冻结验证结果续用。按 v30/v31 的既有处理方式，兼容集合收窄成只有
# v32 自己，代价是所有臂都多冷启动一次。
# v33（2026-09-05，EXP-031 S1）：`IBS_BIAS_OMIT_ZERO_REST_CVS` 默认翻成 True ——
# Group-1 的 CustomCVForce 不再注册那 K 个恒零的 `cv_k_rest` 占位 CV。
# **采样哈密顿量在数学上完全不变**：`cv_k_rest` 是零粒子的
# `CustomExternalForce("0")`，对能量与力的贡献严格为 0，摘掉它是精确恒等变形
# （三条独立路径实测 Group-1 能量逐比特相同，见开关处的说明）。
# 但**表达式字符串与 CV 集合变了** ⟹ system XML 变 ⟹ `system_xml_sha256` 变，
# 因此旧的 dual_window_* / ibs_state_* / convergence.json 一律不能续用：
# 它们产自一个 CV 布局不同的 System，`_int_cv_indices` 与 CV 计数都对不上。
# 这与 v30/v31/v32 同类处理：兼容集合收窄成只有 v33 自己，代价是所有臂多冷启动
# 一次。收益是每步省 0.377 ms（真体系整步 1.2206×）。
IBS_BIAS_PROTOCOL_VERSION = 33

# v28/v29 都只改过 warmup 的停止/诊断控制，没有改变 production Hamiltonian、
# f_k 符号约定或生产采样方式，所以那次是把 27/28/29 都放进同一个兼容集合。
# v30/v31/v32/v33 都是真正的断裂点（分别见上面各自的版本说明），缓存兼容集合只留
# 最新版本自己。
IBS_BIAS_CACHE_COMPATIBLE_PROTOCOL_VERSIONS = frozenset((33,))


def _ibs_bias_protocol_version_is_cache_compatible(value: Any) -> bool:
    try:
        return int(value) in IBS_BIAS_CACHE_COMPATIBLE_PROTOCOL_VERSIONS
    except (TypeError, ValueError):
        return False


def _normalize_ibs_protocol_for_cache_compare(manifest: Dict[str, Any]) -> Dict[str, Any]:
    normalized = json.loads(json.dumps(manifest, sort_keys=True, default=str))
    if _ibs_bias_protocol_version_is_cache_compatible(
        normalized.get("ibs_bias_protocol_version")
    ):
        normalized["ibs_bias_protocol_version"] = IBS_BIAS_PROTOCOL_VERSION
    return normalized


def _recover_stage_identity_from_convergence(
    state_path, state, lambdas_coul, lambdas_vdw, stage_type,
):
    """从同窗口的 `convergence.json` 取回 `stage_protocol_key`。

    **不是"缺了就当通过"** —— 是从一份**独立的、同一次生产写下**的记录里把它取回来。
    前提（缺一不可，否则返回 None 保持 fail-closed）：

      · 找得到这个窗口的 `dual_window_<i>_<stage_type>_convergence.json`；
      · 那份记录里确实有 `stage_protocol_key`；
      · **两份记录的 λ 逐位相等** —— 否则那份 convergence 属于另一套布局，
        借过来就等于用另一个系综的身份给旧 PASS 背书。

    返回 `(identity, source_str)`，取不到返回 `(None, None)`。
    """
    try:
        import glob as _g
        import re as _re
        m = _re.search(r"ibs_state_(\w+)_window_(\d+)\.json$", os.path.basename(state_path))
        if not m:
            return None, None
        st, widx = m.group(1), int(m.group(2))
        ck_dir = os.path.dirname(state_path)
        run_dir = os.path.dirname(os.path.normpath(ck_dir))
        # 段 checkpoint 在 checkpoints/segment_N ⟹ 再上一层才是 run 根。
        if os.path.basename(ck_dir).startswith("segment_"):
            run_dir = os.path.dirname(os.path.dirname(os.path.normpath(ck_dir)))
        pat = f"dual_window_{widx}_{stage_type or st}_convergence.json"
        cands = [c for c in _g.glob(os.path.join(run_dir, "*", pat))
                 if os.path.isfile(c)]
        for c in sorted(cands, key=os.path.getmtime, reverse=True):
            with open(c, encoding="utf-8") as fh:
                conv = json.load(fh)
            key = conv.get("stage_protocol_key")
            if key is None:
                continue
            want = [float(x) for x in (lambdas_vdw or [])]
            got = [float(x) for x in (conv.get("lambdas_vdw") or [])]
            if not want or len(want) != len(got):
                continue
            if any(round(a, LAMBDA_GRID_DECIMALS) != round(b, LAMBDA_GRID_DECIMALS)
                   for a, b in zip(want, got)):
                continue          # 另一套布局的记录，不能借
            return key, os.path.relpath(c, run_dir)
    except Exception:  # noqa: BLE001 —— 恢复失败就保持 None，照旧重走预热
        return None, None
    return None, None


def _resume_cached_window_gate_status(
    cached_conv: Optional[Dict[str, Any]],
    cached_e_shape: Optional[Tuple[int, ...]],
    lc_win: Any,
    lv_win: Any,
    repair_policy: str,
    lse_log_residual_tolerance: float,
    enable_early_stop: bool,
    current_early_stop_config: Dict[str, Any],
    effective_target_steps: int,
    current_coion_identity: Optional[Dict[str, Any]] = None,
    stage_type: str = "coul",
    current_sampling_score_sha256: Optional[str] = None,
    current_stage_protocol_key: Optional[Dict] = None,
) -> Dict[str, Any]:
    """一份磁盘窗口缓存在 resume 时能不能直接复用：10 个门的纯函数求值。

    原先这 ~110 行判断内联在 ``run_all_windows`` 里，依赖 8 个局部变量，没有任何
    办法单独调用，因此"缺字段是否真的 fail-closed""步数预算门是否真的不管
    early_stop_triggered 都生效"这类问题只能靠读代码确认，不能靠测试保证。抽成
    纯函数后每个门都能用 mock 的 ``cached_conv`` 单独验证（见
    ``test_resume_reuse_contracts.py``），且不碰任何文件 I/O。

    刻意留在 ``run_all_windows`` 里没有搬进来的两件事：
      - ``ExistingEnsembleRequiresRescueAudit``：那是带 diagnostics 的 raise，
        不是布尔门，且在 triplet 加载之前就要先拦（保护旧 ensemble 不被覆盖）。
      - ``_load_validated_window_data_triplet`` 的文件读取与 manifest 校验。
    本函数只吃已经解析好的 ``cached_conv`` dict 和能量矩阵的 shape。

    返回 dict 而不是 ``(bool, reason)``：调用方那串逐门诊断打印（每个门有各自的
    文案和一个 legacy_mutating 分支）需要读到**单个门**的布尔值，返回聚合结果会
    迫使那些打印重写。这里把逐门布尔一并返回，调用侧的 elif 阶梯得以一字不动。

    ``usable`` 的与链顺序与原内联判断完全一致，并额外绑定本窗口实际采样的
    co-ion runtime identity 与联合评分身份：
        shape -> lambdas -> wca -> ibs_bias -> lse -> lrc -> repair_policy
        -> coion_identity -> score_identity -> early_stop
    任何字段缺失（旧格式缓存读不到该 key）一律保守判"不可复用"，绝不"大概率
    没变"就放过。
    """
    if not isinstance(cached_conv, dict):
        cached_conv = {}

    # 🔑 形状：ndim==2、物理态数与本次窗口的 λ 数一致、至少有一帧。
    shape_ok = bool(
        cached_e_shape is not None
        and len(tuple(cached_e_shape)) == 2
        and int(tuple(cached_e_shape)[0]) == len(lc_win)
        and int(tuple(cached_e_shape)[1]) > 0
    )

    # 🔑 只按 window_idx + shape 判断"这份缓存还能不能用"是不安全的：λ 路径被
    # 自动加密/重新划分窗口后，同一个 window_idx 完全可能对应一段全新的 λ 区间，
    # 只要凑巧态数相同就会被静默当成有效缓存复用——采样对了，但对的是错的 λ。
    # 旧格式缓存没有这两个字段的，保守地当作不匹配，强制重采。
    cached_lc = cached_conv.get("lambdas_coul")
    cached_lv = cached_conv.get("lambdas_vdw")
    lambdas_match = bool(
        cached_lc is not None
        and cached_lv is not None
        and len(cached_lc) == len(lc_win)
        and len(cached_lv) == len(lv_win)
        and np.allclose(cached_lc, lc_win, atol=1e-9)
        and np.allclose(cached_lv, lv_win, atol=1e-9)
    )

    # 🔑 [wca_accounting_version] λ 值对得上不代表数值对得上：base/bias 的力组
    # 切分口径变过（Group4 λ-WCA 从"算作 base"改成"算作 bias"）时，旧文件里的
    # base/bias 数值是在旧口径下算出来的，即使 λ 完全相同也绝不能复用。
    version_match = cached_conv.get("wca_accounting_version") == WCA_ACCOUNTING_VERSION

    # 🔑 [ibs_bias_protocol_version] 旧协议下这份能量文件可能是在 f_k 全程漂移
    # （生产阶段仍在 update_weights()）的情况下采出来的，不满足 MBAR 的单一固定
    # 采样分布假设——即使 λ/WCA 口径都对得上，也不能当成有效缓存复用。
    bias_protocol_match = _ibs_bias_protocol_version_is_cache_compatible(
        cached_conv.get("ibs_bias_protocol_version")
    )

    # 🔑 [ligand_com_restraint_protocol_version] v1 的 Group 5 配体 COM 限制力在
    # CUDA 上产生永久激活、跨边界跳变的定向拖拽（锚点用未折叠像、centroid 被折叠，
    # 非周期绝对距离把两个周期像当成真实距离），使轨迹不满足 MBAR 的平衡采样前提。
    # 见 4W53/STAGE2_GROUP5_CUDA_PBC_ROOT_CAUSE_2026-08-29.md 与
    # LIGAND_COM_RESTRAINT_PROTOCOL_VERSION 定义处。
    #
    # ⚠️ 这道门不能只挂在 stage 级指纹上：stage 结果因 v1→v2 失效后会重新进入
    # run_all_windows，而**逐窗口**的 energies/bias/base 复用发生在建 Context 之前，
    # 走的是这个门。少了它，旧 Group5 窗口的能量数组仍会被直接复用。
    # 该力只在没有 Boresch 锚定时添加（溶剂腿 stage1/stage2 都有），所以这道门对
    # coul 与 vdw **都**生效，不做 stage_type 豁免。
    # 缺该字段（v1 及更早的缓存根本不写）一律判不可复用。
    com_restraint_version_match = (
        cached_conv.get("ligand_com_restraint_protocol_version")
        == LIGAND_COM_RESTRAINT_PROTOCOL_VERSION
    )

    cached_lse_tolerance = cached_conv.get("lse_log_residual_tolerance")
    lse_tolerance_match = bool(
        cached_lse_tolerance is not None
        and np.isclose(
            float(cached_lse_tolerance),
            float(lse_log_residual_tolerance),
            rtol=0.0,
            atol=1.0e-15,
        )
    )

    # 🔑 [lj_tail_lrc_protocol_version] LRC 公式版本必须匹配——v1 只补 cutoff 外的
    # r^-6、忽略 switching 区间，v2 是真正 switching+softcore-aware 的积分；旧协议
    # 下算出的 target_energies 里叠加的 LRC 数值跟当前公式完全不是同一回事。缺失
    # 该字段（比 v1 更旧）同样保守地判不匹配。
    lrc_version_match = (
        cached_conv.get("lj_tail_lrc_protocol_version")
        == TRADITIONAL_LJ_LRC_PROTOCOL_VERSION
    )

    # 🔑 [vdw_nonbonded_protocol_version，MEM-00h] softcore cutoff/switching 只
    # 影响 Stage 2 vdW window（构造软核 CV 的那条路径），跟 Stage 1 charging
    # 完全无关——charging 阶段不建软核 CV，SOFTCORE_CUTOFF_NM 对它没有意义。
    # 所以这道门只在 stage_type=="vdw" 时才生效；stage_type=="coul" 一律
    # pass-through（True），不因为这个字段缺失就把充电阶段的缓存判废。
    vdw_nb_version_match = (
        True
        if stage_type != "vdw"
        else cached_conv.get("vdw_nonbonded_protocol_version")
        == VDW_NONBONDED_PROTOCOL_VERSION
    )

    # 🔑 [non_mutating_v1] 采样修复策略必须匹配：旧的变异策略缓存（其 f_k 可能被
    # fixed-H 累计 ΔF 就地覆盖过，属于不同参考系）绝不能被非变异策略的 run 复用。
    # 旧缓存没有这个字段（None），与 "non_mutating_v1" 不相等，因此自动判无效。
    # 🔑 [path_evolution_v1] 按**等价类**比，不按字面值：path_evolution_v1 与
    # non_mutating_v1 产出的窗口轨迹同样干净（都没就地改过 f_k），插点时不该
    # 因为策略名变了就把前面所有已完成窗口判废。legacy_mutating 仍自成一类。
    try:
        repair_policy_match = (
            repair_policy_cache_class(str(cached_conv.get("sampling_repair_policy")))
            == repair_policy_cache_class(repair_policy)
        )
    except ValueError:
        repair_policy_match = False   # 缺字段/旧值/拼错一律判不可复用

    # EXP-030: completed-window reuse happens before IBSSampler.load_ibs_state,
    # so the immutable score-family identity must be checked at this early gate.
    sampling_score_match = cached_conv.get("sampling_score_sha256") == current_sampling_score_sha256
    # A matching score hash is not enough if the runtime contract itself was
    # changed.  Baseline caches intentionally stay compatible when no score
    # identity is in use.
    residual_protocol_match = (
        current_sampling_score_sha256 is None
        or cached_conv.get("residual_sampling_protocol_version")
        == IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION
    )

    # 能量/偏置/base 三件套来自具体 co-ion Hamiltonian 的采样分布；仅检查
    # checkpoint manifest 太晚，因为完整能量文件会在建 Context 前被提前复用。
    # neutral 的旧缓存没有该字段，None == None 保持兼容；任何当前 identity
    # 非 None 的运行都要求 convergence.json 有完全相等的 payload。
    cached_coion_identity = cached_conv.get("coion_identity")
    coion_identity_match = cached_coion_identity == current_coion_identity

    # 🔑 [early_stop / 步数预算] 两条独立检查，不管这份缓存是否触发过 early stop
    # 都要过第一条：
    #  (1) 缓存产出时的目标步数不能低于当前调用的目标步数——否则即使从来没有触发
    #      过 early stop、是"完整跑满"的缓存，也可能只是"250k 时代"跑完的窗口，
    #      被静默当成满足"预算已提到 500k"的当前要求复用，实际样本量根本不够。
    #  (2) 如果这份缓存确实是 early_stop_triggered=True 提前停止产出的短样本，还
    #      要求当前调用同样启用 early stop、协议版本一致、且这次实际使用的八个
    #      阈值跟缓存记录的完全一致——只比协议版本不够：版本号只在判据逻辑本身
    #      变了才会变，把某个阈值从松调紧（例如 max_uncertainty_kJ_mol 从 5.0 收紧
    #      到 1.0）根本不影响协议版本，但旧窗口是在更松的阈值下通过的。
    #
    # 🔑🔑 [2026-09-14 / 阶梯空转] 上面 (1) 和 (2) 现在是**两个独立的布尔**。
    # 它们原来挤在同一个 `early_stop_ok` 里（(1) 是 `if`、(2) 是 `elif`），后果有二：
    #   · 归因错：(1) 是**纯进度量**（"目标从 250k 抬到 500k 了"），跟 early stop
    #     毫无关系，却让调用侧打出"缓存是 early stop 提前停止产出的短样本"这种
    #     与现场无关的话（enable_early_stop 默认 False，用户那份缓存的
    #     early_stop_triggered 几乎必然是 False）。
    #   · 更贵的那条：`usable=False` 是**唯一**的否定出口，于是"身份全对、只是
    #     步数没到"被和"身份不符"同等处理 —— 整条前置链（EM 20000 步 + 阶段 2 dt
    #     测试 + Boresch 爬坡 + 冻结 burn-in + 只读复验）每轮重跑一遍，而紧接着
    #     `_try_load_main_window_checkpoint` 又把 Context 的坐标/速度/盒子/RNG
    #     覆盖回上次生产结束那一刻 —— 那条链的动力学产物**逐字被丢弃**。
    # 拆开之后 `usable` / `reason` 的**取值逐字不变**（`step_budget_ok and
    # early_stop_ok` 就是原来那一项，reason 阶梯顺序也照旧 budget 优先 ⟹ 步数
    # 不足的窗口永远不会被跳过），只是多给出第三档 `incremental_topup_only`：
    # 身份全对、只差步数 ⟹ 可以走增量续采，不必重建前置链。
    # `early_stop_ok` 从此**只表示 early-stop 身份**，不再兼职进度判据 —— 名字
    # 捆着两件事正是那句误导性 WARN 的根因，留着下一个人还会被带偏第三次。
    #
    # ⚠️ **目标步数不在身份键里、也不该进去**：`_stage_window_sampling_identity`
    # 显式 `config.pop("n_steps_per_window")`，就是为了把它留在这道门里当**进度**
    # 判据。删了它 750k 的缓存会被当成满足 1000k 目标整窗跳过，抬预算等于白抬。
    cached_target_steps = cached_conv.get("n_steps_per_window_effective")
    step_budget_ok = bool(
        cached_target_steps is not None
        and int(cached_target_steps) >= int(effective_target_steps)
    )
    step_budget_reject_reason: Optional[str] = None
    if not step_budget_ok:
        step_budget_reject_reason = (
            f"当前目标步数（{int(effective_target_steps)}）高于缓存产出时的目标"
            f"（{cached_target_steps}）"
        )

    early_stop_ok = True
    early_stop_reject_reason: Optional[str] = None
    if bool(cached_conv.get("early_stop_triggered", False)):
        if not enable_early_stop:
            early_stop_ok = False
            early_stop_reject_reason = "当前调用未启用 early stop"
        elif (
            cached_conv.get("early_stop_protocol_version")
            != EARLY_STOP_PROTOCOL_VERSION
        ):
            early_stop_ok = False
            early_stop_reject_reason = (
                f"缓存的 early_stop_protocol_version="
                f"{cached_conv.get('early_stop_protocol_version')!r}（期望 "
                f"{EARLY_STOP_PROTOCOL_VERSION}）"
            )
        elif not _early_stop_configs_match(
            cached_conv.get("early_stop_config"), current_early_stop_config
        ):
            early_stop_ok = False
            early_stop_reject_reason = (
                f"缓存的 early_stop_config（{cached_conv.get('early_stop_config')}）"
                f"与当前调用（{current_early_stop_config}）不一致"
            )

    # ⚠️ `early_stop_ok` 从 2026-09-14 起**只表示 early-stop 身份**，不再捎带步数
    # 预算。步数看 `step_budget_ok`。`usable` 取两者的与 ⟹ 取值一个 bit 都没变。
    segment_metadata_match = False
    if shape_ok and cached_conv.get("production_segment_protocol_version") == PRODUCTION_SEGMENT_PROTOCOL_VERSION:
        try:
            _validate_production_segments(cached_conv.get("production_segments"), cached_e_shape[1])
            segment_metadata_match = True
        except (ValueError, TypeError):
            pass
    stage_protocol_match = (
        current_stage_protocol_key is None
        or _stage_window_sampling_identity(cached_conv.get("stage_protocol_key"))
        == _stage_window_sampling_identity(current_stage_protocol_key)
    )
    usable = bool(
        segment_metadata_match
        and stage_protocol_match
        and shape_ok
        and lambdas_match
        and version_match
        and bias_protocol_match
        and com_restraint_version_match
        and lse_tolerance_match
        and lrc_version_match
        and vdw_nb_version_match
        and repair_policy_match
        and sampling_score_match
        and residual_protocol_match
        and coion_identity_match
        # 拆分前这里是一个把两件事捆在一起的 early_stop_ok；取值完全相同。
        and step_budget_ok
        and early_stop_ok
    )

    # 🔑 第三档：**身份全对、只差步数**。`usable` 一字未动（step_budget_ok 仍在
    # 它的与链里），这一档只是把"为什么不可用"分出来，让调用侧能走增量续采而不是
    # 把整条前置链重建一遍。注意它与 `usable` **互斥**：step_budget_ok 为真时它
    # 必为假。
    incremental_topup_only = bool(
        (not step_budget_ok)
        and early_stop_ok
        and segment_metadata_match
        and stage_protocol_match
        and shape_ok
        and lambdas_match
        and version_match
        and bias_protocol_match
        and com_restraint_version_match
        and lse_tolerance_match
        and lrc_version_match
        and vdw_nb_version_match
        and repair_policy_match
        and sampling_score_match
        and residual_protocol_match
        and coion_identity_match
    )

    # reason 的判定顺序与调用侧那串诊断打印的 elif 阶梯一致，保证"日志说的原因"
    # 和"函数报告的原因"永远是同一个门。
    if usable:
        reason: Optional[str] = None
    elif not shape_ok:
        reason = (
            f"缓存能量形状 {tuple(cached_e_shape) if cached_e_shape is not None else None} "
            f"与期望 ({len(lc_win)}, N) 不符"
        )
    elif not version_match:
        reason = (
            f"wca_accounting_version={cached_conv.get('wca_accounting_version')!r}"
            f"（期望 {WCA_ACCOUNTING_VERSION}）"
        )
    elif not bias_protocol_match:
        reason = (
            f"ibs_bias_protocol_version="
            f"{cached_conv.get('ibs_bias_protocol_version')!r}（兼容版本 "
            f"{sorted(IBS_BIAS_CACHE_COMPATIBLE_PROTOCOL_VERSIONS)}）"
        )
    elif not com_restraint_version_match:
        reason = (
            f"ligand_com_restraint_protocol_version="
            f"{cached_conv.get('ligand_com_restraint_protocol_version')!r}"
            f"（期望 {LIGAND_COM_RESTRAINT_PROTOCOL_VERSION}）；v1 的 Group 5 COM "
            "限制力在 CUDA 上产生定向拖拽，该窗口轨迹不满足平衡采样前提"
        )
    elif not lse_tolerance_match:
        reason = (
            f"LSE log 残差容差={cached_lse_tolerance!r}"
            f"（当前 {lse_log_residual_tolerance}）"
        )
    elif not lrc_version_match:
        reason = (
            f"lj_tail_lrc_protocol_version="
            f"{cached_conv.get('lj_tail_lrc_protocol_version')!r}"
            f"（期望 {TRADITIONAL_LJ_LRC_PROTOCOL_VERSION}）"
        )
    elif not vdw_nb_version_match:
        reason = (
            f"vdw_nonbonded_protocol_version="
            f"{cached_conv.get('vdw_nonbonded_protocol_version')!r}"
            f"（期望 {VDW_NONBONDED_PROTOCOL_VERSION}，stage_type={stage_type!r}）"
        )
    elif not repair_policy_match:
        reason = (
            f"sampling_repair_policy="
            f"{cached_conv.get('sampling_repair_policy')!r}（期望 {repair_policy!r}）"
        )
    elif not sampling_score_match:
        reason = (
            "sampling_score_sha256 mismatch: "
            f"cached={cached_conv.get('sampling_score_sha256')!r}, "
            f"current={current_sampling_score_sha256!r}"
        )
    elif not residual_protocol_match:
        reason = (
            "residual_sampling_protocol_version mismatch: "
            f"cached={cached_conv.get('residual_sampling_protocol_version')!r}, "
            f"current={IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION!r}"
        )
    elif not coion_identity_match:
        reason = (
            "co-ion runtime identity 不匹配或缺失："
            f"缓存={cached_coion_identity!r}，当前={current_coion_identity!r}"
        )
    elif not segment_metadata_match:
        reason = "production segment metadata 缺失、版本错误或帧范围不完整"
    elif not stage_protocol_match:
        reason = "stage_protocol_key 与当前采样身份不匹配"
    elif not step_budget_ok:
        # 拆分前这一档和下一档共用同一个 elif（budget 优先）；顺序保持不变，
        # 所以 `reason` 的取值与拆分前逐字相同。
        reason = step_budget_reject_reason
    elif not early_stop_ok:
        reason = early_stop_reject_reason
    else:
        reason = "λ 值不匹配（缺少 λ 元数据或 λ 路径已变更）"

    return {
        "usable": usable,
        "reason": reason,
        "segment_metadata_match": segment_metadata_match,
        "stage_protocol_match": stage_protocol_match,
        "shape_ok": shape_ok,
        "lambdas_match": lambdas_match,
        "version_match": version_match,
        "bias_protocol_match": bias_protocol_match,
        "com_restraint_version_match": com_restraint_version_match,
        "lse_tolerance_match": lse_tolerance_match,
        "cached_lse_tolerance": cached_lse_tolerance,
        "lrc_version_match": lrc_version_match,
        "vdw_nb_version_match": vdw_nb_version_match,
        "repair_policy_match": repair_policy_match,
        "sampling_score_match": sampling_score_match,
        "residual_protocol_match": residual_protocol_match,
        "coion_identity_match": coion_identity_match,
        "cached_coion_identity": cached_coion_identity,
        # [2026-09-14] `early_stop_ok` 现在**只**表示 early-stop 身份；
        # 纯进度短缺看 `step_budget_ok`，两者的与才是原来那一项。
        "early_stop_ok": early_stop_ok,
        "early_stop_reject_reason": early_stop_reject_reason,
        "step_budget_ok": step_budget_ok,
        "step_budget_reject_reason": step_budget_reject_reason,
        "cached_target_steps": cached_target_steps,
        "incremental_topup_only": incremental_topup_only,
    }


# A minibatch stores a full KxN energy matrix plus bias/base vectors.  Keeping
# every learning batch forever makes long/debug warmups grow without bound and
# bloats JSON checkpoints.  The normal production budget is far below this
# limit; the cap only affects unusually long or repeatedly resumed learning.
TMBAR_HISTORY_MAX_ENTRIES = 200

# Warm-up updater identity is deliberately separate from the production bias
# protocol. v9 replaces raw-minibatch SGD control with the absolute physical
# free-energy candidate from the full accumulated TMBAR history. Each update
# is f <- f + 0.2*(f_TMBAR-f). Dominant-state identity is diagnostic only and
# never changes the update. If TMBAR is temporarily unavailable, one bounded
# occupancy step provides coverage acquisition: 4-kT pairwise radius, widened
# to the 10-kT severe ceiling only for a true collapse (raw_residual >= 70). Warm-up frames are
# collected every 250 MD steps (production output cadence is untouched).
# Completed production made by an older updater remains valid; only an
# unfinished older learning checkpoint must not be resumed into this solver.
# [Candidate-first, Validate-or-Learn v1] 9 -> 10：LEARN 不再调用
# update_weights()/全历史 TMBAR 选择器，改为纯 bounded-occupancy 反馈；冻结
# 时机也从"固定 min_bias_updates 批 + 连续通过"改成"raw residual 一旦
# ≤ IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW 立即冻结"。语义变化足以让旧协议下
# 未完成的 in-flight learning 缓存被新逻辑误读，必须让它们版本失配后
# 老实重新预热，不能被当成新语义下的中间状态注入。
IBS_WARMUP_UPDATE_PROTOCOL_VERSION = 10
# 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 0.20 -> 0.10：低重叠时绝对 TMBAR 候选给出错误
# 大步 → 占据塌缩 → TMBAR 更不可靠的正反馈循环。配合下面 IBS_MAX_APPLIED_
# PAIRWISE_STEP_KT 的硬步长上限，一起把每次更新的相邻步长压住。
IBS_TMBAR_UPDATE_DAMPING = 0.10
# 这是 _bounded_log_occupancy_update 的【真塌陷档】pairwise 上限，只在
# raw_residual ≥ IBS_UPDATE_ADAPTIVE_RESIDUAL_COLLAPSE 时才用得到；6≤residual<
# COLLAPSE 的中等区固定 4 kT（见该函数 severe 分支）。
IBS_TMBAR_FALLBACK_SGD_PAIRWISE_STEP_KT = 10.0
IBS_WARMUP_FRAME_STRIDE_STEPS = 250
IBS_TMBAR_LEARNING_MINIBATCH_FRAMES = 40
IBS_TMBAR_FREEZE_MAX_APPLIED_PAIRWISE_STEP_KT = 1.0
# 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 任何一次 warmup 权重更新（绝对 TMBAR 或
# bounded occupancy fallback）应用到 Context 之前，都把相邻步长
# max_k(Δf)-min_k(Δf) 硬 cap 到这个值（2 kT≈5 kJ/mol@300K）。防止低重叠 MBAR
# 候选的巨大 pairwise 跳变一步打崩占据。真实 ΔF>5 kJ 的边靠多轮小步逼近。
IBS_MAX_APPLIED_PAIRWISE_STEP_KT = 2.0
# 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 绝对 TMBAR 候选"可信"门：只有 update_weights
# 里累计 TMBAR 解的质量同时满足以下全部时，才用 _damped_tmbar_absolute_update
# 应用绝对候选；否则退回 _bounded_log_occupancy_update（受限占据反馈，先恢复
# 覆盖）。这些只门控"用哪个更新控制器"，不是窗口收敛判据（收敛仍是 run_all_
# windows 里的 local-MBAR loose gate）。
IBS_TMBAR_TRUST_MIN_OVERLAP = 0.05
IBS_TMBAR_TRUST_MIN_ABSOLUTE_ESS = 10.0
IBS_TMBAR_TRUST_MIN_DECORRELATED_SAMPLES = 10
IBS_TMBAR_TRUST_MAX_UNCERTAINTY_KJ_MOL = 5.0
IBS_TMBAR_TRUST_MIN_COVERAGE_ESS_FRACTION = 0.8
# 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 生产入口门是纯 Δf−ΔF<阈值：abs_ess / min_ess_
# ratio / 占据平坦度 / coverage_ESS 全部只作诊断，不参与放行（曾经用 abs_ess≥10 门
# 控，会把宽松门偷偷变成严格收敛门，已删除）。零重叠外推出的巨大 ΔF（如 908
# kJ/mol）自然因 >阈值被拒，不需要 ESS 门。下面的 OCC_* 常量只供
# _diagnose_local_mbar_situation 判断占据平坦/塌陷（诊断 + 预算耗尽接受时的判决），
# 不参与生产入口放行。
# 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 占据平坦度阈值，仅供 _diagnose_local_mbar_
# situation 的诊断（占据是否平坦 / 是否塌陷）——用于预算耗尽 best-effort 接受时的
# 判决（平坦=良性、塌陷=建议插 λ）以及日志现场，不参与生产入口放行（入口门是纯
# Δf−ΔF<阈值）。平坦定义：每个态占据落在 [OCC_MIN_FRACTION/K, OCC_MAX_FRACTION/K]
# 且 coverage_ESS≥OCC_MIN_COVERAGE_ESS_FRACTION×K。两侧 box 防止
# [0.499,0.499,0.001,0.001] 这类"max<0.5 但塌了两个态"被误判成平坦。
IBS_LOCAL_MBAR_GATE_OCC_MIN_FRACTION = 0.5
IBS_LOCAL_MBAR_GATE_OCC_MAX_FRACTION = 2.0
IBS_LOCAL_MBAR_GATE_OCC_MIN_COVERAGE_ESS_FRACTION = 0.8
# 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] "占据真的塌陷"（硬热力学瓶颈 → 建议插 λ/拆窗）
# 的判据，故意比"未达平坦兜底门"严得多：只有 coverage_ESS 掉到 0.5×K 以下、或某态
# 占据低于 0.25/K（目标的四分之一）才算塌陷。防止把 [0.41,0.27,0.20,0.125] 这类
# 仅仅"最低态贴着 0.5/K 下限"的健康窗口误报成需要插 λ 的瓶颈。塌陷/平坦之间是
# "统计薄，延长采样"的中间态。
IBS_LOCAL_MBAR_GATE_OCC_COLLAPSE_COVERAGE_ESS_FRACTION = 0.5
IBS_LOCAL_MBAR_GATE_OCC_COLLAPSE_MIN_FRACTION = 0.25
# 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] learning→freeze 的 freeze 时机（纯效率、不阻
# 放行）。生产入口门本身是纯 Δf−ΔF<10（见 run_all_windows 里的门）；这里只决定
# "何时值得冻结去跑一次 local-MBAR 检查"，避免在 f_k 还在大步移动时白花 burn-in/
# 验证开销（那正是塌陷窗口浪费预算的原因）。判据只看【应用的 f_k pairwise 步长已
# 降到 ≤ IBS_TMBAR_FREEZE_MAX_APPLIED_PAIRWISE_STEP_KT kT（f_k 几乎不再动）】，连续
# IBS_LEARNING_READY_CONSECUTIVE 批满足才冻结。故意不看占据平坦/coverage/abs_ess
# ——那些会把宽松门偷偷变成严格收敛门。step 门不会阻挡任何本该通过的窗口：
# gap<10 ⟹ 占据残差小 ⟹ bounded 步长小；step 仍大的窗口必然 gap≫10、冻结也白费。
IBS_LEARNING_READY_CONSECUTIVE = 2
IBS_UPDATE_NEAR_FLAT_RELAXATION_FACTOR = 0.35
IBS_UPDATE_MIDDLE_RELAXATION_FACTOR = 0.50
IBS_UPDATE_SEVERE_RELAXATION_FACTOR = 1.00
IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW = 1.0
IBS_UPDATE_ADAPTIVE_RESIDUAL_HIGH = 6.0
# 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] severe 区再分两档的门槛。原来 severe 区的半径
# 从 4 kT 连续爬到 severe_max(10 kT)、residual=12 就到顶，结果 residual 十几到几十
# 的"中等塌陷"窗口也在吃 24.94 kJ/mol 的大步 → dominant 来回翻转、占据震荡。
# 现在：residual < COLLAPSE 一律 4 kT（≈9.98 kJ/mol，单次相对更新不超过 ~10
# kJ/mol）；只有 residual ≥ COLLAPSE 的真塌陷（某态平均 softmax 权重 ~e^-70，
# 靠 4 kT/步要爬十几二十步）才放开到 severe_max。
IBS_UPDATE_ADAPTIVE_RESIDUAL_COLLAPSE = 70.0

# 🔑 [IBS_BIAS_PROTOCOL_VERSION=28] 局部滑窗 MBAR loose-gate。冻结 f_k 后，把最近
# IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES 批固定-f_k minibatch 拼成一次单参考 local
# MBAR（_solve_single_window_local_mbar，跟最终阶段拼接同一套增广矩阵数学，但只看
# 当前冻结 f_k 下最近这几批数据，不吃全历史时变 TMBAR），只比较相邻态 ΔF^MBAR 与
# 当前相邻 Δf_k（gauge 无关，绝对 f_k 的任意公共常数在相邻差里抵消），设唯一 loose
# gate：max_k |Δf_{k,k+1} − ΔF^MBAR_{k,k+1}| < 阈值即冻结进生产。10 kJ/mol ≈ 4 kT，
# 只作"别让某个局部边完全饿死"的粗门；真正的自由能/ESS/overlap/误差全部交给生产后
# 独立的 _assert_stage_result_sane + 最终 MBAR。故意不设连续通过、不设 LSE 占据门、
# 不等 f_k 稳定、不设 warmup ESS 四联门——见 run_all_windows 里的收敛状态机。
IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL = 10.0
IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES = 5
# 🔑 [2026-09-11] 数据不足时的**分级求解**预算。第一次在 SLIDING_BATCHES 批尝试；
# 若是 insufficient_frames*（根本没求出 Δf−ΔF），保持同一份冻结 f_k、保留已有批次
# 继续累计，到 2×/3× 批再试一次，MAX_BATCHES 是单个冻结周期的硬上限。它同时受
# warmup 总步数 full_bias_step_budget 约束（循环条件本身就在数 steps_at_full_bias，
# freeze burn-in 也计在内），哪个先到算哪个；多次冻结不会各自再赠送一份总预算。
#
# ⚠️ 15 是**工程预算，不是"统计充分"的阈值**。到上限仍然样本不足叫**无法判定**：
# 既没有理由改 f_k，也没有依据插 λ。见 IBSValidationBudgetIndeterminateError。
IBS_LOCAL_MBAR_GATE_MAX_BATCHES = 15

# local-MBAR 验证门的 error 分类。三类后果完全不同，混在一起就是把"没测出来"
# 当成"f_k 错了"——那正是插 λ 补救被误触发的入口。
# local-MBAR 门要求的**去相关**帧数下限，也就是可达性预检里的 T。
# ⚠️ 与 `minimum_complete_validation_frames`（原始帧完整性要求，实测 200）是
# **两个量**。混掉会把 gcrit 算小 20 倍 —— 实测后果：win4 只差 26% 帧数
# （600 → 需要 773），却被判成"差 7.5 倍、预算内不可达"。
# λ 比较栅格：与 `lambda_path_versions.LAMBDA_DECIMALS` 同源，**不另立一份**
# （两份常量各自漂移正是这个 bug 的成因）。
try:
    from lambda_path_versions import LAMBDA_DECIMALS as LAMBDA_GRID_DECIMALS
except Exception:  # pragma: no cover —— 极端情况下退回同一个字面量
    LAMBDA_GRID_DECIMALS = 8

IBS_LOCAL_MBAR_GATE_MIN_FRAMES = 10

_LOCAL_MBAR_INSUFFICIENT_DATA_ERRORS = frozenset((
    "insufficient_frames",
    "insufficient_frames_after_decorrelation",
))


def classify_local_mbar_gate_error(error) -> str:
    """``'solved'`` / ``'insufficient_data'`` / ``'unsolvable'`` / ``'input_identity'``。

    · ``solved``：解出了 Δf−ΔF（过不过门是另一回事）。
    · ``insufficient_data``：去相关之后帧数不够，**没测出来**。继续采样可能有救。
    · ``unsolvable``：解了但 f 是 NaN/形状不对。算 f_k/重叠的负面信号，退回 learning。
    · ``input_identity``：形状不一致、能量含 NaN、sampled_row≠0 之类的结构性错误。
      继续采样不会让它消失，必须明着报错，不能当成"数据不足"。
    """
    if error is None:
        return "solved"
    text = str(error)
    if text in _LOCAL_MBAR_INSUFFICIENT_DATA_ERRORS:
        return "insufficient_data"
    if text == "nan_or_shape_mismatch_in_local_mbar_f":
        return "unsolvable"
    return "input_identity"

# 落盘格式的独立版本号：只管 fixed-H 探针轨迹库 checkpoint/manifest 的文件
# 结构（不是采样/校准协议本身），见 probe_adjacent_path_overlap_bank 等函数。
#   version 2: v1 的续采 checkpoint 只存 NPZ（positions/velocities/box），重建
#              Context 后积分器随机数种子从头重设，内部 RNG 状态并未真正恢复——
#              仍被当成同一条不间断轨迹续采，但严格说是"位置/速度对上、但
#              积分器内部状态是全新的"伪续采。改为每个态额外存一份 OpenMM 原生
#              二进制 Context checkpoint（state_{k}_openmm.chk，含积分器内部
#              状态），作为续采的主路径；NPZ 降级为 checkpoint 缺失/损坏/平台
#              不兼容时的兜底——降级到 NPZ 时必须强制开新 segment、重新烧
#              burn-in，不能再当作连续轨迹。见 _resume_or_start_state_simulation。
#   version 3：同上面 IBS_BIAS_PROTOCOL_VERSION=13 的修复——探针轨迹库
#              （_build_fixed_h_probe_bank_manifest 管的 per-state 原始采样
#              轨迹）之前没有对齐 lambda_boresch_scale=1.0，manifest 里也没有
#              任何 code_sha256/代码指纹字段能自动感知这个 Context 层面的
#              修复（common_system_xml 的序列化字节没变，Boresch 力在 XML 里
#              仍然是默认值 0.0，只是运行时 setParameter 覆盖了它）。升版本号
#              让 _fixed_h_probe_bank_manifest_matches 判定所有旧轨迹库整体
#              失配，_invalidate_fixed_h_probe_bank 整目录清空重采。
FIXED_H_PROBE_CACHE_PROTOCOL_VERSION = 3

# 🔑 独立开关：learning_to_validation_cycles>=2 时提前跳出 SGD 收敛循环、直接
# 走 fixed-H overlap 探针+bias 校准兜底，是跟本次轨迹库性能重构同时提交、但
# 逻辑上完全独立的一次策略变更（见 run_all_windows 里对这个常量的唯一读取
# 点）。分开做成开关，是为了以后如果发现行为异常，能立刻二分排查是"轨迹库"
# 还是"早触发"导致的，而不必回退整个提交。默认开启。
IBS_EARLY_PROBE_TRIGGER_ENABLED = True

# 🔑 冻结校准验证阶梯的累计目标步数（50k→150k→300k），single source of truth：
# abfe_pipeline.py::_run_stage_with_overlap_autorepair 导入这个常量而不是自己
# 再定义一份，避免两处硬编码的 schedule tuple 未来改一处漏改另一处。同时也是
# run_all_windows 内部"调用方未提供覆盖字典时"的阶梯档位回退依据——见
# effective_frozen_validation_budget/is_final_rung 的计算处注释。
FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS = (50_000, 150_000, 300_000)

# 🔑 [2026-09-11] warmup 预算账本。三个概念**分开**，别再让一个布尔同时管三件事：
#   · resume_frozen_validation：要不要恢复同一份 f_k + 验证数据、暂不重新学习/校准。
#     "暂不重校准"是为了**把对当前候选的测量做完**，不是认可这个候选。
#   · f_k_evidence_status：这份候选目前有什么证据（见 F_K_EVIDENCE_*）。
#   · warmup_budget_ledger：实际累计消耗 + 当前档位的**累计上限**，决定本次还能跑多少步。
#
# 预算一律是**累计上限**语义：当前上限 B、已消耗 S ⟹ 本次可用 max(0, B − S)。
# 显式升档到 B₂ 才有新增可用 max(0, B₂ − S)。**普通 resume 既不升档也不刷新额度**
# —— 绝不写成 S + 默认 warmup_steps（那是每次 resume 白送一份预算）。
# learning / freeze burn-in / frozen validation 三项分开记账，但共同受同一个总上限约束。
WARMUP_LEDGER_BUCKETS = ("learning_steps", "freeze_burn_in_steps", "frozen_validation_steps")

F_K_EVIDENCE_NONE = "none"                  # 还没有冻结候选
F_K_EVIDENCE_INDETERMINATE = "indeterminate"  # 有候选，但 Δf−ΔF 没测出来 ⟹ 无结论
F_K_EVIDENCE_CALIBRATED = "calibrated"      # legacy MBAR 校准路径给过的"已校准"证据
F_K_EVIDENCE_REFUTED = "refuted"            # 测出来了，被判否
F_K_EVIDENCE_VERIFIED = "verified"          # 通过 loose gate


def new_warmup_budget_ledger(cumulative_cap_steps: int = 0) -> Dict:
    return {
        "learning_steps": 0,
        "freeze_burn_in_steps": 0,
        "frozen_validation_steps": 0,
        "cumulative_cap_steps": int(cumulative_cap_steps),
        "budget_rung": None,
        # False ⟹ 账本是从缺累计步数的旧 checkpoint 迁移来的，消耗可能被低估。
        # 这种账本**不得**用来论证升档。
        "complete": True,
    }


def warmup_ledger_total_steps(ledger: Optional[Dict]) -> int:
    if not ledger:
        return 0
    return sum(int(ledger.get(k, 0) or 0) for k in WARMUP_LEDGER_BUCKETS)


def warmup_ledger_remaining_steps(ledger: Optional[Dict]) -> int:
    """本次还能跑多少步 = max(0, 累计上限 − 已消耗)。"""
    if not ledger:
        return 0
    return max(
        0,
        int(ledger.get("cumulative_cap_steps", 0) or 0)
        - warmup_ledger_total_steps(ledger),
    )


def frozen_candidate_fingerprint(f_k) -> Optional[str]:
    """当前冻结候选的指纹。验证批次额度绑定它：换了 f_k 可以重建批次记录，
    但账本里的**总消耗永远不清零**。

    🔑 **f_k 有 gauge 自由度，必须先规范化再哈希。** 混合权重
    ``∝ exp(−(F_k − f_k)/kT)`` 在归一化之后与 f_k 的公共常数无关，所以
    ``f_k`` 与 ``f_k + c`` 是**同一个采样分布、同一个候选**。逐位哈希会把它们算成
    两个候选，白白重建一次批次记录（并让 :15519 那处"指纹是否仍匹配"的判断误判）。
    减去均值后再哈希；``round(...) + 0.0`` 把 -0.0 归一成 0.0，与
    ``lambda_path_versions._q`` 同一口径。
    """
    if f_k is None:
        return None
    vals = [float(x) for x in f_k]
    if vals:
        mean = sum(vals) / len(vals)
        vals = [round(v - mean, 6) + 0.0 for v in vals]
    payload = ",".join(f"{v:.6f}" for v in vals)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def warmup_ledger_bucket_for_mode(mode: str) -> str:
    if mode == "learning":
        return "learning_steps"
    if mode == "freeze_burn_in":
        return "freeze_burn_in_steps"
    return "frozen_validation_steps"


def base_checkpoint_dir_for(checkpoint_dir: str, max_levels: int = 3) -> Optional[str]:
    """从一个**子命名空间**的 checkpoint 目录找回基准 checkpoint 目录。

    stage-2 有多种子命名空间，形状不一样，别只认一种：
        多采样段   ``<base>/segment_{N}``                （`_recalibrate_fk_and_resample_segment`）
        生产 rescue ``<base>/vanishing_rescue/{plan_id}`` （两层）

    锚点用 ``path_current.json`` —— λ 路径是**全局**的，只有基准目录有它。
    自己就是基准目录（或往上 ``max_levels`` 层都找不到锚点）时返回 None。
    """
    cur = os.path.normpath(checkpoint_dir)
    if os.path.exists(os.path.join(cur, "path_current.json")):
        return None  # 自己就是基准目录，没有"上一层"可继承
    for _ in range(int(max_levels)):
        parent = os.path.dirname(cur)
        if not parent or parent == cur:
            return None
        if os.path.exists(os.path.join(parent, "path_current.json")):
            return parent
        cur = parent
    return None


def inherit_warmup_ledger_across_segments(
    checkpoint_dir: str,
    stage_type: str,
    window_idx: int,
    default_cap_steps: int,
) -> Optional[Dict]:
    """在 checkpoint 子命名空间里开工时，**继承基准目录那一份 warmup 消耗**。

    不是子命名空间、或基准目录里没有这个窗口的 ibs_state，就返回 None。

    🔑 [2026-09-11] 修的是一个实测到的预算漏洞。多采样段有自己的 checkpoint
    命名空间（``checkpoints/segment_2/``），段内 ``ibs_state`` 从零开始，于是
    ``warmup_budget_ledger`` 也是空的 —— 每开一个段就**白送一整份完整额度**。

    实测（cyclod_ligand2/rep1_evidence 的 window 4，cap=555000）::

        vanishing    learning 110k + burn-in 15k + validation 350k = 475k，剩  80k
        vanishing_2  learning  10k + burn-in  5k + validation  50k =  65k，剩 490k
                                                     ← 实际跨段烧了 540k

    这与两处既有约定直接矛盾：``migrate_warmup_budget_ledger`` 的
    "绝不默认『已耗 0 步』再按默认上限赠送一整份额度"，以及
    ``frozen_candidate_fingerprint`` 的"账本里的**总消耗永远不清零**"。

    **继承什么**：只继承三个消耗桶（learning / freeze burn-in / frozen validation）。
    **不继承** f_k / 冻结候选 / ``frozen_validation_cumulative_steps`` —— 新段是
    **新的 f_k Epoch**，那份候选的验证进度本来就该从 0 开始（实测段 2 的该字段
    确实是 0，那是对的）。

    上限用**本次**的 ``default_cap_steps``（预算政策属于本次运行），所以语义是
    "这个窗口**终身** warmup 消耗 ≤ B"。段 2 因此可能只剩很少甚至 0 步 —— 那是
    刻意的：要更多就走显式升档（``_explicit_frozen_validation_rung`` /
    ``FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS``），不靠开新段偷额度。
    """
    base = base_checkpoint_dir_for(checkpoint_dir)
    if base is None:
        return None
    prior_path = os.path.join(base, f"ibs_state_{stage_type}_window_{window_idx}.json")
    try:
        with open(prior_path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, ValueError):
        # 🔑 [BUD-07] **基准段没有这个窗口，不等于它没烧过预算。**
        # 布局变更之后新出现的窗口只在后面的段里跑过 —— 先前这里直接 `return None`，
        # 调用方于是给它**一整份全新额度**，正是本函数要堵的那个洞的另一半。
        # 兄弟段里有它的账本就照常继承；确实哪儿都没有才 `return None`。
        state = None
    if state is None:
        _fallback = _scan_sibling_segment_ledgers(
            base, checkpoint_dir, stage_type, window_idx)
        if _fallback is None:
            return None
        _spent0, _complete0, _srcs0 = _fallback
        ledger = new_warmup_budget_ledger(int(default_cap_steps))
        for bucket in WARMUP_LEDGER_BUCKETS:
            ledger[bucket] = _spent0[bucket]
        ledger["complete"] = _complete0
        ledger["inherited_from"] = _srcs0[0] if _srcs0 else os.path.basename(base)
        ledger["inherited_scanned"] = _srcs0
        return ledger
    prior_ledger = state.get("warmup_budget_ledger")
    if not isinstance(prior_ledger, dict):
        # 基准目录连账本都没有 ⟹ 消耗不可知。**不能**当成 0，走既有的迁移路径，
        # 它会把 complete 标成 False（那种账本不得用来论证升档）。
        return migrate_warmup_budget_ledger(state, int(default_cap_steps))

    # 🔑🔑 [BUD-07，2026-09-14] **只继承基准段是漏的：中间那些段烧的不算数。**
    #
    # `base_checkpoint_dir_for()` 的锚点是 `path_current.json`，而那个文件**只有
    # 基准目录有**。所以 `checkpoints/segment_3` 找回来的永远是 `checkpoints/`，
    # **不是 `segment_2`** —— 第 3 个 Epoch 继承的是基准段的消耗，段 2 烧掉的那一份
    # 凭空消失。链越长漏得越多，而"连开多个 Epoch"正是自治循环修一个难窗口的常态。
    #
    # 形状与已归档的「多 Epoch 链永远从基准段重学 f_k」一模一样（那条的结论是
    # 「第一次换 Epoch 完全正确，连换两次才踩到」），这里是同一个盲区的预算版。
    #
    # 修法：把基准段与**全部兄弟段**的账本一起扫，逐桶取 **max**。
    # 取 max 而不是求和：每个段本来就是「继承上一份再往上加」，所以各段的桶是
    # 同一条单调曲线上的采样点，求和会重复计数；max 既不重复也不会少算，
    # 而且与扫描顺序无关（目录名排序不可靠）。
    # 语义仍然是 docstring 说的那条：**这个窗口终身 warmup 消耗 ≤ B**。
    _spent = {b: int(prior_ledger.get(b, 0) or 0) for b in WARMUP_LEDGER_BUCKETS}
    _complete = bool(prior_ledger.get("complete", True))
    _sources = [os.path.basename(base)]
    _sib = _scan_sibling_segment_ledgers(
        base, checkpoint_dir, stage_type, window_idx)
    if _sib is not None:
        _sib_spent, _sib_complete, _sib_srcs = _sib
        for b in WARMUP_LEDGER_BUCKETS:
            if _sib_spent[b] > _spent[b]:
                _spent[b] = _sib_spent[b]
        _complete = _complete and _sib_complete
        _sources.extend(s for s in _sib_srcs if s not in _sources)

    ledger = new_warmup_budget_ledger(int(default_cap_steps))
    for bucket in WARMUP_LEDGER_BUCKETS:
        ledger[bucket] = _spent[bucket]
    ledger["complete"] = _complete
    ledger["inherited_from"] = os.path.basename(base)
    # 逐桶最大值实际取自哪几个段 —— 漏账再发生时这一行就是证据。
    ledger["inherited_scanned"] = _sources
    return ledger


def _scan_sibling_segment_ledgers(base, checkpoint_dir, stage_type, window_idx):
    """扫基准目录下**全部兄弟段**里这个窗口的 warmup 账本，逐桶取 max。

    🔑🔑 [BUD-07，2026-09-14] `base_checkpoint_dir_for()` 的锚点是
    ``path_current.json``，而那个文件**只有基准目录有**。所以
    ``checkpoints/segment_3`` 找回来的永远是 ``checkpoints/``、**不是
    ``segment_2``** —— 第 3 个 Epoch 继承的是基准段的消耗，段 2 烧掉的那一份凭空
    消失。链越长漏得越多，而"连开多个 Epoch"正是自治循环修一个难窗口的常态。
    形状与已归档的「多 Epoch 链永远从基准段重学 f_k」一模一样（那条的结论是
    「第一次换 Epoch 完全正确，连换两次才踩到」），这里是同一个盲区的预算版。

    取 **max** 而不是求和：每个段本来就是「继承上一份再往上加」，各段的桶是同一条
    单调曲线上的采样点，求和会重复计数；max 既不重复也不会少算，而且与扫描顺序
    无关（目录名排序不可靠）。

    返回 ``(spent_by_bucket, complete, sources)``；一个兄弟段都没读到返回 None。
    ⚠️ 兄弟段有 ``ibs_state`` 却没有账本 ⟹ 它的消耗**不可知**，``complete=False``；
    不得当成 0，否则一份少算过的账会被拿去论证"还能升档"。
    """
    spent = {b: 0 for b in WARMUP_LEDGER_BUCKETS}
    complete, sources, hit_any = True, [], False
    me = os.path.normpath(checkpoint_dir)
    try:
        # ⚠️ 本模块**没有** import glob —— 用 os.listdir，别在这里引新依赖：
        # 这段外面裹着 except，NameError 会被静默吞掉、悄悄退回漏账的旧行为。
        names = sorted(n for n in os.listdir(base) if n.startswith("segment_"))
    except OSError:
        return None
    for name in names:
        sib = os.path.join(base, name)
        if not os.path.isdir(sib) or os.path.normpath(sib) == me:
            continue
        try:
            with open(os.path.join(
                    sib, f"ibs_state_{stage_type}_window_{window_idx}.json"),
                    "r", encoding="utf-8") as fh:
                sst = json.load(fh)
        except (OSError, ValueError):
            continue
        led = sst.get("warmup_budget_ledger")
        if not isinstance(led, dict):
            complete = False
            continue
        hit_any = True
        touched = False
        for b in WARMUP_LEDGER_BUCKETS:
            v = int(led.get(b, 0) or 0)
            if v > spent[b]:
                spent[b], touched = v, True
        complete = complete and bool(led.get("complete", True))
        if touched:
            sources.append(name)
    if not hit_any and complete:
        return None
    return spent, complete, sources


def migrate_warmup_budget_ledger(state: Dict, default_cap_steps: int) -> Dict:
    """旧 checkpoint 没有账本时，**优先从已保存的运行记录恢复**，恢复不了就标记不完整。

    绝不默认"已耗 0 步"再按默认上限赠送一整份额度。能恢复的只有
    ``frozen_validation_cumulative_steps``（v12 起就持久化的那份冻结验证累计步数）；
    learning / burn-in 的历史消耗没有任何持久记录，所以只要走到迁移这一步，
    ``complete`` 一律为 False，并在诊断里如实标出来。
    """
    ledger = new_warmup_budget_ledger(default_cap_steps)
    recovered = int(state.get("frozen_validation_cumulative_steps", 0) or 0)
    ledger["frozen_validation_steps"] = recovered
    ledger["complete"] = False
    ledger["migrated_from"] = (
        "frozen_validation_cumulative_steps" if recovered else "nothing_recoverable"
    )
    return ledger


def _resolve_frozen_validation_budget_for_window(
    window_idx,
    frozen_validation_step_overrides,
    prior_cumulative_steps: int,
    schedule=FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS,
) -> int:
    """Pick this attempt's cumulative step-budget target for a window.

    Prefers the caller's explicit per-window override (the normal case: a
    same-process ladder escalation in abfe_pipeline.py). Falls back to the
    smallest schedule rung strictly greater than the already-persisted
    ``prior_cumulative_steps`` when no override is present -- this is the
    case right after a process restart (--resume), where the caller's
    in-memory override dict is empty but the window's real progress already
    lives on disk. Falling back to schedule[0] unconditionally there would
    silently discard that progress (see run_all_windows call site).
    """
    if window_idx in (frozen_validation_step_overrides or {}):
        return int(frozen_validation_step_overrides[window_idx])
    return next(
        (rung for rung in schedule if rung > int(prior_cumulative_steps)),
        schedule[-1],
    )


def _resolve_frozen_validation_is_final_rung(
    window_idx,
    frozen_validation_is_final_rung,
    effective_frozen_validation_budget: int,
    schedule=FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS,
) -> bool:
    """Whether this attempt's budget target is the ladder's last rung.

    Prefers the caller's explicit per-window flag; falls back to comparing
    the resolved budget against the schedule's last entry when the caller
    has no entry for this window (post-restart, same reasoning as
    _resolve_frozen_validation_budget_for_window -- must not default to
    False, or a window whose persisted progress already reached the final
    rung would never be judged terminal and would loop on the same rung.
    """
    if window_idx in (frozen_validation_is_final_rung or {}):
        return bool(frozen_validation_is_final_rung[window_idx])
    return bool(int(effective_frozen_validation_budget) >= schedule[-1])


class IBSWarmupConvergenceError(IBSControlPlaneError):
    """Structured signal that a window lacks adequate sampled-state coverage."""

    def __init__(self, message: str, diagnostics: Dict):
        super().__init__(message)
        self.diagnostics = diagnostics


def validation_reachability_verdict(
    g_measurements,
    *,
    required_decorrelated_frames,
    cycle_frame_cap,
    budget_remaining_steps,
    frame_stride_steps=IBS_WARMUP_FRAME_STRIDE_STEPS,
    frames_already=0,
):
    """冻结验证在**算术上**还有没有可能凑够去相关帧数。

    ⚠️ **只改路由，不改 verdict。** 不可达是**预算**问题，绝不等于 f_k 被驳回：
    `UNMEASURED` 永远不许因为这个被改写成 `REJECTED`。也不许仅凭高 g 去插 λ
    或拆窗 —— 证据类型到动作集合是固定映射，不得跨类回答。

    判据（T = 要求的去相关帧数，Ncap = 本周期最多原始帧）::

        gcrit_cycle  = Ncap / T
        gcrit_budget = (已有帧 + 剩余预算 / 每帧步数) / T

    `g_L` 取**多个检查点里的最小值**，不是 bootstrap：同一候选上的连续检查点是
    **嵌套**的（400 帧包含 200 帧），bootstrap 的独立性前提不成立、会低估方差，
    那样算出来的"保守下界"是假的。取最小值才是嵌套数据下真正的保守下界。

    方向很重要：只有当**乐观下界都超阈值**（`g_L > gcrit_budget`）时才敢判不可达，
    所以不会因为 g 在小样本下的一次涨落误杀一份好 f_k。
    """
    gs = [float(x) for x in (g_measurements or []) if x is not None and float(x) > 0]
    T = float(required_decorrelated_frames)
    if T <= 0 or not gs:
        return {"verdict": "INDETERMINATE", "reason": "no_g_measurements"}
    stride = max(1, int(frame_stride_steps))
    budget_frames = float(frames_already) + float(budget_remaining_steps) / stride
    gcrit_cycle = float(cycle_frame_cap) / T
    gcrit_budget = budget_frames / T
    g_lo, g_hi = min(gs), max(gs)
    out = {
        "g_measurements": gs,
        "g_lower_bound": g_lo,
        "g_upper": g_hi,
        "g_lower_bound_method": "min_over_checkpoints_nested_samples",
        "required_decorrelated_frames": T,
        "cycle_frame_cap": float(cycle_frame_cap),
        "budget_frames_available": budget_frames,
        "gcrit_cycle": gcrit_cycle,
        "gcrit_budget": gcrit_budget,
        "projected_raw_frames_needed": T * g_hi,
        "projected_steps_needed": T * g_hi * stride,
        "may_be_used_to_reject_f_k": False,
    }
    if len(gs) < 2:
        out.update(verdict="INDETERMINATE",
                   reason="need_at_least_two_checkpoints_on_same_candidate")
        return out
    if g_lo > gcrit_budget:
        # 乐观下界都不够 ⟹ 这份候选在剩余预算内**不可能**验完。
        out.update(verdict="UNREACHABLE", reason="g_lower_bound_exceeds_budget_gcrit")
        return out
    if g_hi <= gcrit_cycle:
        out.update(verdict="REACHABLE_THIS_CYCLE", reason="g_upper_within_cycle_gcrit")
        return out
    if g_hi <= gcrit_budget:
        out.update(verdict="REACHABLE_WITHIN_BUDGET",
                   reason="cycle_too_short_but_budget_may_suffice")
        return out
    # 区间跨过阈值：最多再给**一个** +250k 诊断块，之后必须裁决，不许再循环。
    out.update(verdict="INDETERMINATE", reason="uncertainty_straddles_gcrit_budget")
    return out


def sealed_candidate_matches(f_k_new, f_k_sealed, *, tol_kJ_mol=0.5):
    """新候选是不是**实质上还是被驳回的那一份**（mean-center 后逐态比较）。

    不能用 fingerprint 相等来判：fingerprint 是哈希，fresh LEARN 出来的 f_k 在
    浮点上永不会逐比特相同，精确相等这条护栏**永远不触发**。真正要挡的是"换了
    一份但其实还是同一份 f_k"，那是 f_k 空间里的**距离**问题。

    容差复用仓库已有的 `min_adjacent_shift_kJ_mol` 口径（0.5 kJ/mol）——
    同一个常量在别处的含义就是"f_k 到底动没动"，不另造阈值。
    """
    a = [float(x) for x in (f_k_new or [])]
    b = [float(x) for x in (f_k_sealed or [])]
    if not a or len(a) != len(b):
        return False
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    return max(abs((x - ma) - (y - mb)) for x, y in zip(a, b)) <= float(tol_kJ_mol)


def refuted_candidate_identity(f_k, window_idx) -> Dict[str, Any]:
    """被驳回候选的**封存身份**：f_k 向量 + 溯源指纹 + 窗口。**三个键名的唯一出处。**

    🔑🔑 [2026-09-14 / S1] 封存路径此前是断的：`abfe_pipeline` 捕到
    `IBSFrozenCalibrationValidationError` 后按 `candidate_fingerprint` /
    `frozen_f_k_kJ_mol` / `f_k` 三个键取身份，而**这三个键在两个 raise 点上一个都
    不存在**（terminal raise 传的是 4 键 dict；另一个传 `bias_warmup_diag`，那里
    的向量叫 `frozen_f_k_at_last_freeze`）。结果每条封存记录都是
    `"f_k_kJ_mol": []` + 指纹 null，而 `sealed_candidate_matches` 开头就是
    `if not a or len(a) != len(b): return False` ⟹ **空身份永远匹配不上任何新
    候选**，"永不续验"这条护栏在运行代码里等于不存在。

    修法是**两侧统一到一组名字**（老板 2026-09-14 拍板），而不是在 raise 点迁就
    调用方再造新名字：
      · `frozen_f_k_kJ_mol`     —— f_k 向量本身，**这就是封存身份**
        （`seal_refuted_candidate` 的 docstring：语义身份，不是哈希）。选这个名字
        而不是 `frozen_f_k_at_last_freeze`：带单位、描述**内容**而不是时机，而且
        `IBSValidationBudgetIndeterminateError` 的诊断里**已经**叫这个名字 ⟹
        两族异常从此同名同义，不再制造"同名不同源"。
      · `candidate_fingerprint` —— **仅溯源，不参与判定**（自产哈希不配做身份，
        这坑本仓已复发四次）。
      · `window_index`          —— 被驳回的窗口下标。
    异常的 `diagnostics` 是封存记录的**唯一身份来源**，所以每个 raise 点都必须
    把这三个键带全 —— 用这个函数生成，别手写。

    ⚠️ `frozen_f_k_at_last_freeze` **并存保留**（同一个向量写两遍）：它在
    `ibs_engine` 的 convergence 交叉核对（见本文件 ~545 行）和
    `tests/test_analysis_f_k_is_the_production_entry_one.py` 里都有活读点。
    `frozen_f_k_kJ_mol` 是权威名，另一个是历史名。
    """
    vals = None if f_k is None else [float(x) for x in f_k]
    return {
        "window_index": int(window_idx),
        "frozen_f_k_kJ_mol": vals,
        "candidate_fingerprint": frozen_candidate_fingerprint(vals),
        # 历史名，有活读点所以并存；权威名是 frozen_f_k_kJ_mol。
        "frozen_f_k_at_last_freeze": vals,
    }


def assert_candidate_not_sealed(
    f_k,
    *,
    checkpoint_dir: Optional[str],
    window_idx: int,
    lambdas_vdw,
    stage_type: str,
) -> None:
    """新候选若与**已封存**（被统计驳回）的候选实质相同，立刻按"这份已被驳回"处理。

    🔑🔑 [2026-09-14 / S1] `sealed_candidate_matches` 此前**全仓零生产调用点**
    （只有定义 + 一个孤立测数学的单测）。真正拦住重试的只有 `relearn_epoch_used`
    （按 `(path_version, window)` 计数）⟹ 实际执行的是"每个窗口只许再试一次"，
    **f_k 距离那层护栏根本不在运行代码里** —— 正是设计注释点名要防的"反复试到
    偶然通过"。这个函数就是把它接进生产路径：**所有进入冻结验证的候选都要过这道门。**

    匹配口径：同一 `window_idx` + 同一 λ 身份（`lambdas_vdw`，与
    `seal_refuted_candidate` 一样 round 到 1e-10）下，f_k 向量 mean-center 后逐态
    距离 ≤ 0.5 kJ/mol。**故意不比 `path_version`**：ibs_engine 这一层拿不到它，
    而 λ 身份是更强的判据（λ 表变了 λ 身份就变了；λ 没变的话换个 path_version
    也还是同一份物理候选，本来就不该放行）。

    驳回时抛 `IBSFrozenCalibrationValidationError(terminal=True)` —— 不新造异常
    类型：这条异常的语义（"已被驳回、不再自动续验/延长预算/回退 learning"）
    就是这里要表达的东西，上层的处置路径也已经是对的那条。
    """
    if not checkpoint_dir or not f_k:
        return
    from abfe_preoptimizer import read_sealed_candidates  # 惰性：避免循环 import

    lam_id = [round(float(x), 10) for x in (lambdas_vdw or [])]
    cand = [float(x) for x in f_k]
    for entry in read_sealed_candidates(checkpoint_dir):
        if int(entry.get("window_idx", -1)) != int(window_idx):
            continue
        if list(entry.get("lambda_identity") or []) != lam_id:
            continue
        if not sealed_candidate_matches(cand, entry.get("f_k_kJ_mol")):
            continue
        raise IBSFrozenCalibrationValidationError(
            f"窗口 {window_idx}（{stage_type}）的这份候选 f_k 与已封存、被统计驳回的"
            f"候选实质相同（mean-center 后逐态差 ≤ 0.5 kJ/mol，封存理由 "
            f"{entry.get('reason')!r}）⟹ **不许续验**。换一份实质不同的候选，或人工"
            "检查为什么 learning 又收敛到同一个点；反复把同一份 f_k 送去验证，"
            "等的就是它偶然通过。",
            diagnostics={
                "window_index": int(window_idx),
                "stage_type": stage_type,
                "sealed_candidate_rematched": True,
                "sealed_entry_reason": entry.get("reason"),
                "sealed_entry_path_version": entry.get("path_version"),
                **refuted_candidate_identity(cand, window_idx),
            },
            terminal=True,
        )


class IBSValidationBudgetIndeterminateError(IBSControlPlaneError):
    """冻结验证预算耗尽，但一直**样本不足**——Δf−ΔF 从来没被求出来过。

    与 ``IBSWarmupConvergenceError`` 的区别是决定性的，别合并：那个是"测出来了、
    f_k 不对"（⟹ λ 太稀），这个是"**没测出来**，对 f_k 无结论"。所以它**不是**修改
    f_k 或退回 SGD 的理由。

    ⚠️ 它**不触发拆窗**，`_run_stage2_with_path_evolution` 故意不捕获它。
    拆窗的依据只有"f_k 压不平"（IBSWarmupConvergenceError）；去相关后有效帧数
    不够是**采样问题，拆窗治不了**——拆完照样测不出来，只会每轮拆一次直到预算
    耗尽，还白白改掉实验布局。这里的结局就是报"无法判定"，交人决定。

    例外：resume 前就 ``bias_status=converged`` 的窗口（复验路径）不走这条 ——
    已经通过过的结论不能被"没测出来"否定，见 run_all_windows 里那个分支。

    冻结 f_k、已累计的批数和已耗预算都已落盘，resume 接着验。
    """

    def __init__(self, message: str, diagnostics: Dict,
                 window_idx: Optional[int] = None):
        super().__init__(message)
        self.diagnostics = diagnostics
        # 🔑 [2026-09-16] **哪个窗口无结论，必须机器可读。**
        # 先前只有消息字符串里写着窗口号 ⟹ 调用方想"把这个窗口记成 indeterminate、
        # 继续跑其余窗口"就只能去正则那句中文。固定预算模式（自治控制器关闭）下
        # 这条信号没人接管，这个下标就是唯一能把"无结论"和"失败"分开的东西。
        self.window_idx = None if window_idx is None else int(window_idx)


class ExistingEnsembleRequiresRescueAudit(IBSControlPlaneError):
    """On-disk data for this window was produced under a DIFFERENT sampling
    repair policy (e.g. the deprecated mutating path that could rewrite f_k in
    place). Under non_mutating_v1 we refuse to reuse it AND refuse to overwrite
    it — the original files are preserved and this is raised so the rescue audit
    can decide, per ensemble, whether the data is reusable (clean, pure-SGD) or
    must be re-run. Never silently continue and clobber the old ensemble."""

    def __init__(self, message: str, diagnostics: Optional[Dict] = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


# 🔑 [non_mutating_v1] Fail-closed policy gate. ONLY the explicit legacy value
# re-enables the deprecated mutating fixed-H probe + in-place f_k recalibration.
# Any unrecognized value (e.g. a typo) MUST raise — never silently fall back to
# mutation. This is the single decision point used by run_all_windows so the
# behavior is testable without a GPU (see test_non_mutating_policy.py).
# 🔑 [path_evolution_v1 / 2026-09-10] 第三个策略：**路径演化**。
# 与 legacy_mutating 的本质区别：它从不就地重写 f_k、也从不把相邻态 fixed-H
# overlap 当收敛仲裁（那是 ATT-27 移除那 881 行循环的设计错误，别搬回来）。它只在
# f_k 真的调不动时（IBSWarmupConvergenceError）在失败区间插一个 λ，登记成新的路径
# 版本，后面的状态顺延；已完成、λ 逐位未变的窗口照常复用。
# 因此对**窗口轨迹缓存**而言它与 non_mutating_v1 同类：两者产出的 f_k 都没有被
# 就地污染过。只有 legacy_mutating 自成一类。
_VALID_SAMPLING_REPAIR_POLICIES = (
    "non_mutating_v1", "path_evolution_v1", "legacy_mutating",
)
_NON_MUTATING_POLICY_CLASS = frozenset(("non_mutating_v1", "path_evolution_v1"))


def repair_policy_cache_class(repair_policy: str) -> str:
    """窗口轨迹缓存的等价类。同类之间可以互相复用。

    仅用于判断"已采好的窗口能不能复用"，不决定任何执行行为——后者一律走
    `should_run_legacy_repair` / `should_run_path_evolution`。
    """
    if repair_policy in _NON_MUTATING_POLICY_CLASS:
        return "non_mutating"
    if repair_policy == "legacy_mutating":
        return "legacy_mutating"
    raise ValueError(
        f"unknown sampling repair_policy {repair_policy!r}; expected one of "
        f"{_VALID_SAMPLING_REPAIR_POLICIES}."
    )


def should_run_path_evolution(repair_policy: str) -> bool:
    """True only for 'path_evolution_v1'; raise for anything unrecognized."""
    if repair_policy not in _VALID_SAMPLING_REPAIR_POLICIES:
        raise ValueError(
            f"unknown sampling repair_policy {repair_policy!r}; expected one of "
            f"{_VALID_SAMPLING_REPAIR_POLICIES}. Refusing to guess — fail closed."
        )
    return repair_policy == "path_evolution_v1"


def should_run_legacy_repair(repair_policy: str) -> bool:
    """Return True only for the explicit deprecated 'legacy_mutating' policy;
    False for 'non_mutating_v1' / 'path_evolution_v1'; raise for anything else."""
    if repair_policy in _NON_MUTATING_POLICY_CLASS:
        return False
    if repair_policy == "legacy_mutating":
        return True
    raise ValueError(
        f"unknown sampling repair_policy {repair_policy!r}; expected one of "
        f"{_VALID_SAMPLING_REPAIR_POLICIES}. Refusing to guess — fail closed "
        "(a typo must NOT re-open the deprecated mutating fixed-H/recalibration path)."
    )


class IBSFrozenCalibrationValidationError(IBSControlPlaneError):
    """一份已经用 fixed-H overlap 探针 + bias 校准探针证明过物理正确的冻结 f_k，
    在（当前这次或累计）冻结验证预算内仍未通过独立验证——跟
    IBSWarmupConvergenceError（从未获得过一份校准 f_k 的普通未收敛，真正需要
    SGD 继续搜索或拆窗/插 λ）是两种完全不同的失败模式，故意用不同的类型，
    避免上层（abfe_pipeline.py 的拆窗/插 λ 修复逻辑）把两者混为一谈。

    ``terminal=True`` 表示这已经是终态（冻结验证累计预算已经用到调用方标记的
    最后一档仍未通过，对应 bias_status="calibrated_validation_failed"）——
    调用方不应再尝试任何形式的自动续验/延长预算/回退 SGD，需要人工检查。
    ``terminal=False`` 表示这次只是当前这一档预算内没验证完，diagnostics 里的
    calibration_pending_validation=True 供调用方决定是否延长预算重试。
    """

    def __init__(self, message: str, diagnostics: Dict, terminal: bool = False):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.terminal = bool(terminal)


class IBSSampler:
    """
    IBS 采样器：严格通过 CustomCVForce 探针收集各态微扰能。
    【修复】按论文 Sec. 2.3/eq. 13 累计跨迭代相对 Q_k 的 TA 估计。
    【修复】能量收集时自动扣除参考态偏移，确保数值稳定性。
    """
    def __init__(self, context: openmm.Context, n_states: int,
                 temperature: unit.Quantity, prefix: str = "abfe",
                 ibs_wrapper: IBSBiasForce = None):
        self.context = context
        self.n_states = n_states
        self.prefix = prefix
        self.ibs_wrapper = ibs_wrapper
        _temp = temperature if hasattr(temperature, 'value_in_unit') else temperature * unit.kelvin
        self.kt = (unit.MOLAR_GAS_CONSTANT_R * _temp).value_in_unit(unit.kilojoule_per_mole)
        self.beta = 1.0 / self.kt
        
        # EMA 只保留为原始占据趋势诊断，不参与权重更新或 learning 收敛门。
        self.ema_mean_p = None
        self.gamma = 0.9  # 衰减因子，保留 90% 历史信息
        # [IBS_BIAS_PROTOCOL_VERSION=19] 论文 eq. 15 TMBAR 状态：每个 update_weights()/
        # 冻结验证 minibatch 打包成一条 {'u_kn','bias_energies','base_energies',
        # 'lambda_indices'=全部本窗口态,'sampled_distribution_row'=0} entry，永久
        # 追加（不像旧 TA 估计器那样每轮用完就丢）。f_k 由
        # solve_stage_integrated(self.tmbar_history, self.kt) 的逆方差加权拼接
        # 给出——因为每条 entry 的 lambda_indices 完全重叠（都是同一窗口的全部
        # 态，只是采样时刻的 f_k 不同），拼接逻辑天然按每条 entry 自己的采样
        # 分布把它们正确对齐合并，正是"聚合一系列时变 IBS 分布样本"。v23 起
        # 解出的绝对向量不再直接写回 f_k，只用于累计覆盖质量门和诊断。
        self.tmbar_history: List[Dict[str, Any]] = []
        self.tmbar_history_dropped_entries = 0
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=8] 每次 frozen validation 失败、恢复
        # learning 时减半（下限见 apply_learning_rate_penalty），让重新学习的
        # 步子比上一次尝试更保守，而不是用几乎相同的大步长继续来回振荡。
        self.eta_penalty = 1.0
        self.energy_buffer = []
        self.energy_history = []
        self.f_history = []
        self.bias_history = []
        self.base_energy_history = []
        self.last_update_diagnostics = {}
        # Exact Group-1 state energies and residual basis, distinct from the
        # physical target ledger which excludes the sampling-only residual.
        self.sampling_state_energy_history = []
        self.residual_basis_history = []
        # New checkpoints preserve real f trajectories; old ones still load
        # through the update-count fallback below.
        self.sampling_score_family_sha256 = None
        self.last_frozen_batch_size = 0
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=25] dominant-state oscillation watchdog
        # (diagnostic only, does not affect control flow): tracks whether the
        # argmax of mean_p_batch keeps flipping between consecutive
        # update_weights() calls, which is the visible symptom of the
        # multi-state overshoot fixed in this version.
        self._last_dominant_k = None
        
        # 🔑 新增：能量偏移缓存
        self.e_offset = 0.0
        # 连续 Base 能量读取失败计数——单帧失败允许跳过（很可能是瞬时 getState
        # 异常），但连续多帧失败说明 Context/系统本身有问题，必须硬报错，而不是
        # 一直用假 0.0 悄悄污染 base_energy_history。
        self._consecutive_base_failures = 0
        self._energy_query_attempts = 0
        self._energy_query_failures = 0
        self._energy_query_consecutive_failures = 0
        self._energy_query_failure_reasons: Dict[str, int] = {}
        # 🔑 只有在 run_all_windows 的严格收敛判据真正通过之后才置 True；
        # 生产采样阶段以此为准冻结 f_k（不再调用 update_weights），且这个
        # 状态会随 save_ibs_state/load_ibs_state 落盘/恢复，见 IBS_BIAS_PROTOCOL_VERSION。
        self.bias_converged = False
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12] 单独的 bias_converged bool 无法区分
        # "从未校准过、普通未收敛"和"MBAR 校准已经给出正确 f_k，只是冻结验证
        # 还没在预算内通过"——这两者 resume 时必须走不同路径：前者应该回到
        # [learning] 重新 SGD；后者的 f_k 已经是好的，只应该继续 freeze_burn_in/
        # validating，绝不能被当成热启动喂回 SGD 重新调整。bias_status 记录
        # 四态："unconverged"（默认）/"calibrated_pending_validation"/"converged"/
        # "calibrated_validation_failed"（真实 GPU 生产日志发现的补丁：冻结验证
        # 累计预算已经用到调用方标记的最后一档仍未通过，是终态，不再自动续验/
        # 延长预算/回退 SGD——跟"calibrated_pending_validation"的关键区别就是
        # "还要不要继续自动重试"）；frozen_f_k_pending 是 calibrated_pending_
        # validation 状态下被冻结的那份 f_k 快照，随 save_ibs_state/load_ibs_state
        # 落盘/恢复；到达终态后设为 None（不再是"pending"，不该被继续消费）。
        self.bias_status = "unconverged"
        self.frozen_f_k_pending = None
        # 🔑 这份冻结 f_k 累计已经花在冻结验证上的步数（跨越同一份校准 f_k 的
        # 多次 resume/阶梯升级累加，不是单次 attempt 内的 steps_at_full_bias）。
        # 只有在 calibrated_pending_validation 状态下才有意义，随 save_ibs_state/
        # load_ibs_state 落盘/恢复。没有它，每次阶梯升级（50k→150k→300k）都会
        # 把 run_all_windows 内部的 while 循环预算当成"这次要新跑这么多步"而
        # 不是"累计到这么多步"，导致真实总步数变成 50k+150k+300k=500k 而不是
        # 阶梯设计意图的"累计延长到 300k"。
        self.frozen_validation_cumulative_steps = 0
        # 🔑 [2026-09-11] 第五态 "frozen_validation_indeterminate"：冻结验证一直
        # **样本不足**、Δf−ΔF 从未被求出来，单周期批次预算用尽。它跟
        # "calibrated_validation_failed" 的区别是"测出来了 f_k 不对" vs "没测出来"；
        # 跟 "calibrated_pending_validation" 的区别是后者断言这份 f_k 已被证明物理
        # 正确（那是已弃用的变异校准路径才产生的状态，non_mutating 下见到即 fail）。
        # 这一态下 frozen_f_k_pending 存冻结 f_k、frozen_validation_batches_done 存
        # 已经累计到第几批：resume 必须接着验同一份 f_k、接着用同一份批次，
        # **不重新领一份批次额度**。批次数据本身随 tmbar_history 一起落盘。
        self.frozen_validation_batches_done = 0
        # 🔑 [2026-09-11] 三概念拆分之二/之三。bias_status 仍是 resume 分支用的状态机；
        # f_k_evidence_status 单独回答"这份候选有什么证据"，不跟"要不要恢复冻结验证"
        # 混在一起——无法判定态要恢复冻结验证（把测量做完），但**不继承"f_k 已正确"**。
        self.f_k_evidence_status = F_K_EVIDENCE_NONE
        # 累计消耗 + 当前档位累计上限；None 表示还没建（首次 save 时建）。
        self.warmup_budget_ledger = None
        # 验证批次额度绑定**当前冻结候选**：换了 f_k 可以重建批次记录，但账本里的
        # 总消耗永远不清零。存候选指纹用来判断批次记录还属不属于这份 f_k。
        self.frozen_candidate_fingerprint = None
        # 载入的采样身份（旧 PASS 只在同一身份下有效）；没载入过就是 None。
        self.loaded_stage_protocol_key = None
        # 🔑 [2026-08-27，见 docs/experiments/EXP-030_FROZEN_SNAPSHOT_TIMING_BUG_2026-08-26.md]
        # 只在 run_all_windows 真正进入生产采样这一刻被设置一次，随
        # save_ibs_state/load_ibs_state 落盘/恢复。None 表示这份冻结 f_k
        # 还没有被观测到过"第一次进入生产"的时刻——续采时绝不能用当前
        # attempt 重新读到的 Context 值覆盖已恢复的旧值，那会用"现在"冒充
        # "最初"。
        self.production_entry_f_k = None
        # [Candidate-first, Validate-or-Learn v1] 纯附加元数据，不参与
        # run_all_windows 的任何控制流分支。seed_source 由实际生效的 SEED
        # 路径写入：load_ibs_state() 成功即写 "resume"；pilot/bootstrap
        # 热启动在 run_all_windows 里写 "pilot"/"bootstrap"；LEARN 第一次
        # 真正改动 f_k 时改写为 "learned"。
        self.seed_source: Optional[str] = None
        self.validation_attempts = 0
        self.last_failure_reason: Optional[str] = None
        # run_all_windows 在 load 前会用调用方已通过枚举校验的 policy 覆盖此值；
        # 这里仍给直接使用 IBSSampler 的调用方一个 fail-closed 默认值，避免遗漏
        # 赋值时把无策略/旧策略 state 当成可恢复状态注入 Context。
        self.sampling_repair_policy = "non_mutating_v1"
        self._probe_context = None
        self._probe_integrator = None
        self._probe_groups = []
        if self.ibs_wrapper is not None and getattr(self.ibs_wrapper, "_int_cv_force_xmls", None):
            self._build_probe_context()

    def _build_probe_context(self):
        main_system = self.context.getSystem()
        probe_sys = openmm.System()
        for i in range(main_system.getNumParticles()):
            probe_sys.addParticle(main_system.getParticleMass(i))
        # 直接调用、不吞异常——probe context 用来算的 CV 里有基于 PBC 的距离项，
        # 如果这里静默失败，probe_sys 会留着 OpenMM System 的默认盒子（跟
        # main_system 完全不一致），后面 evaluate_interaction_energies() 算出来
        # 的相互作用能就会是错的盒子下的错误值，而它直接喂给在线 IBS bias 更新
        # 和 TMBAR，没有任何信号说这里出过问题。main_system 的盒矢量本身已经在
        # 真实 Context 里跑通，正常情况下这一步不该失败；真失败了就该在这里炸。
        probe_sys.setDefaultPeriodicBoxVectors(*main_system.getDefaultPeriodicBoxVectors())

        # 🔑 [构建前检查] 探针 Context 给每个 λ 态分配一个独立 force group，
        # 起始号 PROBE_FORCE_GROUP_BASE=16，而 OpenMM 的 force group 上限是 31
        # ⟹ 单个探针 Context 最多容纳 32-16 = 16 个 λ 态。
        # 不做这个检查的话，第 17 个态会在 setForceGroup(32) 上抛
        # `OpenMMException: Force group must be between 0 and 31`——那句话既不说明
        # 是哪一层的限制，也不告诉调用方该怎么办。
        # ⚠️ 这是**探针 Context 的**限制，与 build_ibs_dual_system 里那条
        # IBS_DUAL_MAX_LAMBDA_STATES（CustomCVForce 的 32-CV 上限）是两回事：
        # 后者约束 IBS 混合偏置力，独立固定态采样根本不用它。
        n_cv = len(self.ibs_wrapper._int_cv_force_xmls)
        if n_cv > PROBE_MAX_LAMBDA_STATES:
            raise RuntimeError(
                f"探针 Context 无法容纳 {n_cv} 个 λ 态：每态占用一个 force group，"
                f"起始号 {PROBE_FORCE_GROUP_BASE}，OpenMM 上限 "
                f"{OPENMM_MAX_FORCE_GROUP} ⟹ 单次最多 {PROBE_MAX_LAMBDA_STATES} 个态。"
                "请把 λ 区间拆成多段分别求值；不要通过降低 PROBE_FORCE_GROUP_BASE "
                "来腾空间——16 以下的组号被生产 Hamiltonian 的力占用（0/1/2/3/4/5）。"
            )
        self._probe_groups = []
        for idx, force_xml in enumerate(self.ibs_wrapper._int_cv_force_xmls):
            force = openmm.XmlSerializer.deserialize(force_xml)
            gid = PROBE_FORCE_GROUP_BASE + idx
            force.setForceGroup(gid)
            probe_sys.addForce(force)
            self._probe_groups.append(gid)

        self._probe_integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        self._probe_context = openmm.Context(probe_sys, self._probe_integrator, self.context.getPlatform())

    def evaluate_interaction_energies(self, positions, box_vectors=None) -> np.ndarray:
        """Evaluate every target CV at arbitrary coordinates without changing dynamics."""
        if self._probe_context is None or not self._probe_groups:
            return np.zeros(self.n_states, dtype=float)

        self._probe_context.setPositions(positions)
        if box_vectors is not None:
            # 同上：不吞异常。调用方显式传了这一帧的盒子，说明它认为盒子形状变了
            # （NPT/膜 barostat 之后常见），这里静默失败就意味着 probe context
            # 继续用上一次成功设置的旧盒子去算这一帧的相互作用能——这是一个
            # 实实在在喂进采样/统计的物理量，不是诊断边角料，不能猜、不能吞。
            self._probe_context.setPeriodicBoxVectors(*box_vectors)

        interaction_energies = np.zeros(self.n_states, dtype=float)
        for k, gid in enumerate(self._probe_groups[:self.n_states]):
            state = self._probe_context.getState(getEnergy=True, groups={gid})
            interaction_energies[k] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        return interaction_energies

    def _evaluate_interaction_energies_live(self) -> np.ndarray:
        """[性能修复：主 Context 一次读全部 CV] `collect_energies()` 的热路径永远读
        的是 `self.context` **此刻**的活体状态（生产/warmup 主循环紧跟
        `sim.step()` 之后调用，中间没有任何 `setPositions`）——而每个 `cv_k_int`
        早在 `build_ibs_dual_system`/`build_shadow_coul_ibs_system` 里就已经被
        注册成了 `self.ibs_wrapper.force`（Group 1，驱动真实动力学的那个
        CustomCVForce）的 collective variable。既然主 Context 自己就知道全部
        K 个 CV 的值，`getCollectiveVariableValues()` 一次调用就能拿到全部，
        完全不需要 `evaluate_interaction_energies()` 那条路径——重新反序列化
        K 个力、建一个独立 probe System/Context、再逐个 force group 发起
        `getState()`——的 K 次 GPU 同步。

        ⚠️ 只能用在"当前坐标就是 self.context 此刻坐标"的场景。`evaluate_
        interaction_energies(positions, box_vectors)` 仍然是**唯一**给任意
        历史/候选帧（fixed-H probe bank、bidirectional overlap 探针）求值的入口，
        原样保留，不受这次改动影响。
        """
        indices = self.ibs_wrapper._int_cv_indices[: self.n_states]
        cv_vals = self.ibs_wrapper.get_force().getCollectiveVariableValues(self.context)
        return np.asarray([cv_vals[i] for i in indices], dtype=float)

    def _collect_interaction_energies(
        self,
        reuse_positions=None,
        reuse_box_vectors=None,
    ) -> Tuple[np.ndarray, Any]:
        """返回 (interaction_energies, box_vectors)。

        [性能修复：主 Context 一次读全部 CV] 这个方法的两个调用方
        (`collect_energies()`/`get_raw_interaction_energies()`) 在整个仓库里
        从不传显式坐标——读的永远是 `self.context` 此刻的活体状态，也就是说
        `positions` 参数本身从未真正需要被喂给一个独立 probe context（那只是
        旧实现为了"逐 λ 查询"绕的路）。现在直接从主 Context 的 CustomCVForce
        读全部 CV（`_evaluate_interaction_energies_live()`），`reuse_positions`
        参数保留只是为了不用改调用方签名，不再被使用。`reuse_box_vectors` 仍然
        有效：省掉一次只为了拿盒子而单独发起的 `getState()`，返回值同时喂给
        `_lj_tail_correction_kj_mol` 算体积。
        """
        if reuse_box_vectors is not None:
            box_vectors = reuse_box_vectors
        else:
            try:
                box_vectors = self.context.getState().getPeriodicBoxVectors()
            except Exception:
                box_vectors = None
        return self._evaluate_interaction_energies_live(), box_vectors

    def get_raw_interaction_energies(self) -> np.ndarray:
        energies, _ = self._collect_interaction_energies()
        return energies.copy()

    def _lj_tail_correction_kj_mol(self, reuse_box_vectors=None) -> np.ndarray:
        """解析 LJ 长程色散尾项修正（逐态,加到 interaction_energies 上）。
        lj_tail_lrc_coeff_kj_mol（switching+softcore-aware，见
        _lj_tail_lrc_coefficients_kj_mol）由 build_ibs_dual_system 预计算并挂在
        ibs_wrapper 上；这里每帧只需读当前盒子体积做一次除法，不重新积分。
        缺失/不适用 (potential_type='dexp' 或没有周期性盒子) 时返回全 0，
        不影响原有行为。fixed-H overlap 探针（probe_bidirectional_overlap）
        和这里读的是同一个 ibs_wrapper.lj_tail_lrc_coeff_kj_mol，保证两处
        用的是同一组系数。

        [性能修复] `reuse_box_vectors` 允许调用方传入本帧已经取到的 box
        vectors（例如 `_collect_interaction_energies()` 刚返回的那一份），
        省掉一次只为了拿盒子体积就单独发起的 getState() 调用——之前这里
        无条件独立查询一次，跟同一次 collect_energies() 里稍早已经查过的
        盒子完全重复。默认 None 时保持原来独立查询的行为。
        """
        lrc_coeff = getattr(self.ibs_wrapper, "lj_tail_lrc_coeff_kj_mol", None) if self.ibs_wrapper is not None else None
        if lrc_coeff is None:
            return np.zeros(self.n_states, dtype=float)
        try:
            if reuse_box_vectors is not None:
                box_vectors = reuse_box_vectors
            else:
                box_vectors = self.context.getState().getPeriodicBoxVectors()
            if box_vectors is None:
                raise ValueError("LRC 需要周期性盒向量，但当前 Context 未提供")
            box_nm = box_vectors.value_in_unit(unit.nanometer)
            a, b, c = (np.asarray(v, dtype=np.float64) for v in box_nm)
            volume_nm3 = abs(np.dot(a, np.cross(b, c)))
        except Exception as exc:
            raise RuntimeError("LJ 长程尾项体积计算失败，拒绝以零修正继续") from exc
        if not np.isfinite(volume_nm3) or volume_nm3 <= 0.0:
            raise RuntimeError(f"LJ 长程尾项盒体积非法: {volume_nm3!r}")
        coeff = np.asarray(lrc_coeff, dtype=np.float64).ravel()
        if coeff.size != self.n_states:
            raise RuntimeError(
                "LJ 长程尾项系数长度与 sampler 状态数不符："
                f"{coeff.size} != {self.n_states}"
            )
        if not np.all(np.isfinite(coeff)):
            raise RuntimeError("LJ 长程尾项系数含 NaN/Inf")
        return coeff / volume_nm3

    def energy_query_diagnostics(self) -> Dict[str, Any]:
        attempts = int(getattr(self, "_energy_query_attempts", 0))
        failures = int(getattr(self, "_energy_query_failures", 0))
        return {
            "attempts": attempts,
            "failures": failures,
            "failure_fraction": float(failures / attempts) if attempts else 0.0,
            "consecutive_failures": int(
                getattr(self, "_energy_query_consecutive_failures", 0)
            ),
            "failure_reasons": dict(
                getattr(self, "_energy_query_failure_reasons", {})
            ),
            "limits": {
                "max_consecutive_failures": ENERGY_QUERY_MAX_CONSECUTIVE_FAILURES,
                "max_total_failures": ENERGY_QUERY_MAX_TOTAL_FAILURES,
                "max_failure_fraction": ENERGY_QUERY_MAX_FAILURE_FRACTION,
                "fraction_min_attempts": ENERGY_QUERY_FAILURE_FRACTION_MIN_ATTEMPTS,
            },
        }

    def assert_energy_query_quality(self, final: bool = False) -> None:
        diag = self.energy_query_diagnostics()
        attempts = int(diag["attempts"])
        fraction_gate = attempts > 0 and (
            final or attempts >= ENERGY_QUERY_FAILURE_FRACTION_MIN_ATTEMPTS
        )
        if (
            int(diag["consecutive_failures"])
            >= ENERGY_QUERY_MAX_CONSECUTIVE_FAILURES
            or int(diag["failures"]) >= ENERGY_QUERY_MAX_TOTAL_FAILURES
            or (
                fraction_gate
                and float(diag["failure_fraction"])
                > ENERGY_QUERY_MAX_FAILURE_FRACTION
            )
        ):
            raise RuntimeError(
                "IBS 能量查询失败超过 hard gate："
                f"attempts={attempts}, failures={diag['failures']}, "
                f"fraction={diag['failure_fraction']:.3%}, "
                f"consecutive={diag['consecutive_failures']}, "
                f"reasons={diag['failure_reasons']}"
            )

    def _record_energy_query_result(
        self,
        success: bool,
        reason: Optional[str] = None,
    ) -> None:
        if success:
            self._energy_query_consecutive_failures = 0
            return
        self._energy_query_failures = int(
            getattr(self, "_energy_query_failures", 0)
        ) + 1
        self._energy_query_consecutive_failures = int(
            getattr(self, "_energy_query_consecutive_failures", 0)
        ) + 1
        reasons = dict(getattr(self, "_energy_query_failure_reasons", {}))
        key = str(reason or "unknown")
        reasons[key] = int(reasons.get(key, 0)) + 1
        self._energy_query_failure_reasons = reasons
        self.assert_energy_query_quality(final=False)

    def collect_energies(
        self,
        *,
        reuse_positions=None,
        reuse_box_vectors=None,
        reuse_e_base=None,
        reuse_e_bias=None,
    ) -> np.ndarray:
        """[性能修复] `reuse_positions`/`reuse_box_vectors`：生产控制面的
        guard 代码块（见 `run_all_windows` 主循环）在 `do_force_check=True`
        的 update 上已经做过一次 `getState(getPositions=True, ...)`，把结果
        传进来可以省掉这里独立再查一次同一个 context、同一时刻的
        positions/box——两者读到的是完全相同的物理状态，中间没有任何动力学
        演化。默认 None 时保持原来独立查询的行为，其余调用点（冷启动/warmup/
        legacy 校准循环等）不用改。

        [性能修复：合并 guard 查询] `reuse_e_base`/`reuse_e_bias`：guard 弱分支
        （`do_force_check=False`）现在自己直接查 `groups={0,2,3,5}`/`{1,4}`
        （见 `run_all_windows`），因为生产窗口系统（`build_ibs_dual_system`/
        `build_shadow_coul_ibs_system`）的力只落在 Group 0-5，`e_base+e_bias`
        恒等于不过滤的 `getState(getEnergy=True)` 总能量——guard 用这两个数
        既能判"总能量是否有限"，又能把它们喂给这里，省掉这里本来还要再查一遍
        同一个瞬时状态的两次 `getState()`。guard 只有在两者都有限时才会走到
        这个调用（否则在 guard 里就已经 rollback + continue，不会进这个函数），
        所以下面两段"非有限"分支理论上不会触发，仍然保留是为了不静默吞掉任何
        未来误用（例如新调用点传了没检查过的值）。
        """
        self._energy_query_attempts = int(
            getattr(self, "_energy_query_attempts", 0)
        ) + 1
        energies = np.zeros(self.n_states)
        failure_reason = None
        # 1. Base 能量 (Group 0, 2, 3, 5)：严格 λ 无关的物理项。
        #    🔑 [wca_accounting_version=2] Group 4 (λ-WCA 防护壳) 曾经也在这里，被当成
        #    "λ 无关"处理——但它的 lambda_shield 是每个窗口各自设成"本窗口 λ_vdw 均值"
        #    (见 run_all_windows 里 setParameter("lambda_shield", ...))，同一个重叠 λ 态
        #    在两个不同窗口里因此带着不同的 WCA 取值，却被当成同一个物理态拼接——
        #    这是 Hamiltonian 不一致，不只是效率问题。Group 4 现在明确定义为纯采样期
        #    偏置（帮助窗口内动力学不塌缩，跟 Group 1 的 IBS flattening bias 同类），
        #    连同 Group 1 一起在 bias_history 里被完整记录、完整 reweight 掉，不再计入
        #    任何目标态的物理能量。是否把 WCA 改造成"每个目标态各自算一次"的正式中间
        #    Hamiltonian（效率更高但不影响正确性）留作后续独立评估，见 wca_accounting_version。
        if reuse_e_base is not None:
            # guard 已经用 groups={0,2,3,5} 查过这个瞬时状态的 e_base，且只有在
            # 它和 e_bias 都有限时才会走到这个调用（见上面 docstring）——正常
            # 情况下这里恒真。仍然显式检查有限性，不静默信任传入值。
            e_base = float(reuse_e_base)
            if np.isfinite(e_base):
                self._consecutive_base_failures = 0
            else:
                failure_reason = "base_energy"
                self._consecutive_base_failures += 1
                print(
                    f"  [ERR] Base 能量 (Group 0,2,3,5，来自 guard 复用值) 非有限"
                    f"（连续第 {self._consecutive_base_failures} 次），本帧标记为 NaN 并跳过"
                )
        else:
            try:
                state_base = self.context.getState(getEnergy=True, groups=set(IBS_E_BASE_FORCE_GROUPS))
                e_base = state_base.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                self._consecutive_base_failures = 0
            except Exception as e:
                failure_reason = "base_energy"
                # ⚠️ 不能静默吞掉：e_base 会被直接写入 base_energy_history 并喂给
                # GlobalMBARAnalyzer (u_phys_kj = base_kj + u_kj_raw)。之前失败时回退
                # 成 0.0 并"标记"——但下面只检查 interaction_energies 是否有 NaN，e_base
                # 本身从未被检查，假 0.0 会照常被 append 进 base_energy_history，
                # 悄悄污染 MBAR 且无迹可查。现在改为 NaN（下面统一按"任一分量非有限
                # 就整帧跳过"处理，不单独放行 e_base），单帧失败允许跳过；但连续失败
                # 说明 Context/系统本身有问题（不是偶发 getState 抖动），必须硬报错，
                # 而不是无限跳帧、悄悄丢数据。
                self._consecutive_base_failures += 1
                # On the first failure, distinguish a transient energy-query
                # problem from a simulation that already has non-finite
                # coordinates/forces.  In the latter case waiting five more MD
                # chunks would only let a broken trajectory drift further.
                if self._consecutive_base_failures == 1:
                    try:
                        diagnostic_state = self.context.getState(
                            getPositions=True,
                            getForces=True,
                        )
                        diagnostic_positions = np.asarray(
                            diagnostic_state.getPositions(asNumpy=True).value_in_unit(
                                unit.nanometer
                            ),
                            dtype=np.float64,
                        )
                        diagnostic_forces = np.asarray(
                            diagnostic_state.getForces(asNumpy=True).value_in_unit(
                                unit.kilojoule_per_mole / unit.nanometer
                            ),
                            dtype=np.float64,
                        )
                    except Exception as diagnostic_exc:
                        raise RuntimeError(
                            "Base 能量首次读取失败，且坐标/力诊断也无法完成；拒绝继续推进 MD。"
                        ) from diagnostic_exc
                    if (
                        not np.all(np.isfinite(diagnostic_positions))
                        or not np.all(np.isfinite(diagnostic_forces))
                    ):
                        raise RuntimeError(
                            "Base 能量首次读取失败时检测到非有限坐标或力；系统已经失稳，"
                            "拒绝继续推进 MD。"
                        ) from e
                print(
                    f"  [ERR] Base 能量 (Group 0,2,3,5) 获取失败（连续第 "
                    f"{self._consecutive_base_failures} 次），本帧标记为 NaN 并跳过：{e}"
                )
                e_base = float("nan")
        if self.ibs_wrapper is None:
            self._record_energy_query_result(False, "missing_ibs_wrapper")
            return np.full(self.n_states, np.nan)

        try:
            if reuse_e_bias is not None:
                # 同上：guard 已经用 groups={1,4} 查过同一瞬时状态，且只有在它和
                # e_base 都有限时才会走到这里——正常情况下恒真，仍显式检查。
                e_bias = float(reuse_e_bias)
                if not np.isfinite(e_bias):
                    failure_reason = failure_reason or "bias_energy"
            else:
                try:
                    # Group 1 (IBS flattening bias) + Group 4 (λ-WCA 防护壳，纯采样偏置，
                    # 详见上面 collect_energies 顶部注释)：两者都只影响采样分布、不代表任何
                    # 目标态的物理能量，必须一起完整 reweight 掉，缺一个都会重新引入偏差。
                    state_bias = self.context.getState(getEnergy=True, groups=set(IBS_E_BIAS_FORCE_GROUPS))
                    e_bias = state_bias.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                except Exception:
                    e_bias = np.nan
                    failure_reason = failure_reason or "bias_energy"

            # 🔑 [LRC vs 实际施加的 CV 不一致] softcore_energies 是真正驱动 Group1
            # log-sum-exp 偏置力的那个量（build_ibs_dual_system 里的 cv_k_int，纯
            # softcore，不含 LRC）；lrc 修正只是 Python 侧对"物理能量"的解析加成，
            # 从未进过实际的 OpenMM 力。之前两者被合成同一个数组，同时喂给
            # update_weights()（学 f_k，理应对齐"实际施加的偏置力"）和 MBAR（理应
            # 对齐"完整物理能量，含 LRC"）——用同一个含 LRC 的量去训练 f_k，学到的
            # 是一个真实偏置力并不存在的目标，降低 IBS 平坦化效率（不影响 MBAR
            # 正确性：sampled row 用的是直接读出的 e_bias，与 f_k 好坏无关）。
            # 现在分开：bias_cv_energies（纯 softcore，喂 energy_buffer/update_weights）
            # vs target_energies（softcore+LRC，喂 energy_history/MBAR）。
            softcore_energies, interaction_box_vectors = self._collect_interaction_energies(
                reuse_positions=reuse_positions,
                reuse_box_vectors=reuse_box_vectors,
            )
            lrc_energies = self._lj_tail_correction_kj_mol(interaction_box_vectors)
            target_energies = softcore_energies + lrc_energies

            residual_basis_energy = self.ibs_wrapper.get_residual_basis_energy(
                self.context
            )
            if self.ibs_wrapper.residual_enabled:
                residual_coefficients = (
                    self.ibs_wrapper.get_sampling_state_coefficients()
                )
                residual_offset = float(
                    self.ibs_wrapper.residual_energy_offset_kj_mol
                )
                sampling_state_energies = (
                    softcore_energies
                    + residual_coefficients
                    * (residual_basis_energy - residual_offset)
                )
            else:
                sampling_state_energies = softcore_energies.copy()

            # 🔑 [2026-08-26 bug fix] 之前这里用的是纯物理 softcore_energies，
            # 完全不含残差项。但驱动实际采样动力学的 Group-1 偏置力公式是
            # X_k = U_k + s_residual·A_k·B_φ − f_k（见 _state_expr(k)，
            # IBS_BIAS_PROTOCOL_VERSION=30 引入 s_residual 时改的那处）——f_k
            # 在线学习（update_weights()/_solve_tmbar_and_recenter()）读的
            # self.energy_buffer 如果只喂纯物理量，训练目标和实际部署目标完全
            # 对不上：f_k 学的是"如何拉平不含残差的分布"，却被拿去拉平"含残差
            # 的分布"，残差项越大的态天然学不对。已经在候选臂三个独立 repeat
            # 的生产数据里实测到系统性占据分布偏斜（跟这个不对齐完全吻合）。
            # sampling_state_energies（上面已经算出来，含残差、full-strength，
            # 不随 bias_scale/s_residual 爬坡缩放——这是有意的：bias_scale/
            # s_residual 只是部署期的爬坡机制，f_k 训练目标应该始终是满强度的
            # 最终混合分布，物理项这边一直就是这么处理的，没有把 bias_scale
            # 也乘进 u_mk）才是训练该用的目标；baseline 臂 residual_enabled=False
            # 时 sampling_state_energies 就是 softcore_energies 的原样拷贝，
            # 这一改动对 baseline 逐字节不变。
            # 相对偏移防溢出 (以 State 0 为参考)——只用于 bias_cv 训练路径的数值稳定性：
            # softmax/log-sum-exp 对每帧内所有态统一平移不变。
            #
            # ⚠️ [2026-08-31 P1 更正] 这里原来还写着「MBAR 那边存的是未平移的
            # target_energies 原始值」——**那句是错的**。`_append_tmbar_batch_from_buffer()`
            # 取的是 `self.energy_buffer`（即下面这个已减 e_offset 的 bias_cv_energies），
            # 不是 `self.energy_history`（target_energies）。逐帧偏移因此漏进了
            # tmbar_history 的 u_kn，见该函数里的说明。
            if self.n_states > 0 and np.isfinite(sampling_state_energies[0]):
                self.e_offset = sampling_state_energies[0]
            bias_cv_energies = sampling_state_energies - self.e_offset
            energies = bias_cv_energies

            frame_finite = (
                np.all(np.isfinite(bias_cv_energies))
                and np.all(np.isfinite(target_energies))
                and np.isfinite(e_base)
                and np.isfinite(e_bias)
                and np.all(np.isfinite(sampling_state_energies))
                and np.isfinite(residual_basis_energy)
            )
            if frame_finite:
                self.energy_buffer.append(bias_cv_energies)
                self.energy_history.append(target_energies.copy())
                self.base_energy_history.append(float(e_base))
                self.bias_history.append(float(e_bias))
                self.sampling_state_energy_history.append(
                    sampling_state_energies.copy()
                )
                self.residual_basis_history.append(float(residual_basis_energy))
                self._record_energy_query_result(True)
            else:
                self._record_energy_query_result(
                    False,
                    failure_reason or "nonfinite_energy_component",
                )
        except Exception as e:
            if isinstance(e, RuntimeError) and (
                "hard gate" in str(e)
                or "LJ 长程尾项" in str(e)
            ):
                raise
            print(f"  [WARN] CV 探针能量提取失败: {e}")
            energies[:] = np.nan
            self._record_energy_query_result(False, "cv_probe")
        return energies

    def _append_tmbar_batch_from_buffer(self) -> int:
        """[IBS_BIAS_PROTOCOL_VERSION=19] 把当前 self.energy_buffer（清空前）打包成
        一条持久的 tmbar_history entry，供论文 eq. 15 TMBAR 聚合。

        必须在任何清空 self.energy_buffer 的操作（update_weights() 末尾 /
        evaluate_frozen_batch_probability() 末尾）之前调用：collect_energies() 对
        energy_buffer/bias_history/base_energy_history 用同一个 frame_finite 门同步
        append，因此清空前 energy_buffer 的最后 len(energy_buffer) 帧，与
        bias_history/base_energy_history 的对应尾部条目逐帧一一对应。
        返回实际追加的有效（非 NaN）帧数（0 表示这一批没有可用样本）。
        """
        m0 = len(self.energy_buffer)
        if m0 == 0:
            return 0
        # 🔑 [2026-08-31 P1] `u_kn` 必须是**未减逐帧 e_offset** 的能量。
        # `self.energy_buffer` 里装的是 `sampling_state_energies - e_offset`，而
        # `e_offset` 是 `collect_energies()` 里**逐帧**更新的 `sampling_state_energies[0]`。
        # 旧实现把它直接当 u_kn 存进 tmbar_history，却配上**未偏移**的
        # `bias_energies`/`base_energies`：下游 `solve_stage_integrated` /
        # `_solve_single_window_local_mbar` 组增广矩阵时目标行是 `base + u_kn`
        # （含 −c_n）、采样行是 `base + bias`（不含），于是这个逐帧平移不再是
        # 全行公共量、在 MBAR 里不抵消，等于人为注入一个共模因子 c_n。
        # 4W53 output_v2 实测 sd(c_n)/kT = [3.10, 1.98, 1.76, 0.84, 0.33, 0.13]
        # （window 0→5），而本文件 ESS_GATE 注释里记的**真实**防护壳共模只有
        # 0.95~2.40 kT —— 耦合端注入的伪共模比真实物理项还大。
        # `sampling_state_energy_history` 与 `energy_buffer` 在 `collect_energies()`
        # 里受同一个 `frame_finite` 门同步 append（也与 `bias_history`/
        # `base_energy_history` 同步），所以取 `[-m0:]` 尾部与下面 bias/base 的
        # 取法完全同构，逐帧一一对应。
        sampling_tail = self.sampling_state_energy_history[-m0:]
        if len(sampling_tail) != m0:
            raise RuntimeError(
                "tmbar entry 组装失败：sampling_state_energy_history 尾部长度 "
                f"{len(sampling_tail)} 与 energy_buffer 长度 {m0} 不一致，"
                "两者本应受同一个 frame_finite 门同步 append。拒绝用错位数据建 MBAR。"
            )
        u_mk_raw = np.array(sampling_tail, dtype=np.float64)
        valid_mask = ~np.isnan(u_mk_raw).any(axis=1)
        n_valid = int(np.sum(valid_mask))
        if n_valid == 0:
            return 0
        bias_tail = np.asarray(self.bias_history[-m0:], dtype=np.float64)
        base_tail = np.asarray(self.base_energy_history[-m0:], dtype=np.float64)
        self.tmbar_history.append({
            "u_kn": u_mk_raw[valid_mask].T,
            "bias_energies": bias_tail[valid_mask],
            "base_energies": base_tail[valid_mask],
            "lambda_indices": list(range(self.n_states)),
            "sampled_distribution_row": 0,
        })
        overflow = len(self.tmbar_history) - TMBAR_HISTORY_MAX_ENTRIES
        if overflow > 0:
            del self.tmbar_history[:overflow]
            self.tmbar_history_dropped_entries += overflow
        # 滑动窗口：即使 dominant 不切换，累积过多 entry 也会因早期帧权重过大
        # 而拖慢 TMBAR 收敛。只保留最近 N 条（每条约 20 帧），双保险。
        _TMBAR_SLIDING_WINDOW = 10
        if len(self.tmbar_history) > _TMBAR_SLIDING_WINDOW:
            n_drop = len(self.tmbar_history) - _TMBAR_SLIDING_WINDOW
            self.tmbar_history = self.tmbar_history[-_TMBAR_SLIDING_WINDOW:]
            self.tmbar_history_dropped_entries += n_drop
        return n_valid

    def _solve_tmbar_and_recenter(
        self,
        min_ess_ratio: float = 0.05,
        min_absolute_ess: float = 5.0,
        min_decorrelated_samples: int = 5,
        max_uncertainty_kJ_mol: float = 5.0,
        min_frames_per_window: int = 3,
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """[IBS_BIAS_PROTOCOL_VERSION=19] 对 self.tmbar_history 里累计的全部
        minibatch 跑一次论文 eq. 15 TMBAR（复用 GlobalMBARAnalyzer.
        solve_stage_integrated，同一份代码同时服务于生产/最终分析的窗口间拼接），
        取回按本窗口态数 mean-centered 的 f_k。

        阈值默认刻意宽松（"这份候选值不值得冻结去走 frozen validation"），真正
        严格的接受/拒绝判据仍是 run_all_windows 里独立、未变的冻结验证阶段——
        候选门放宽不会让坏 f_k 蒙混过关，只会让它更快地被送去验证、验证失败后
        样本并入 tmbar_history 继续学习。

        min_frames_per_window=3（而不是 solve_stage_integrated 生产/最终分析
        用的默认 10）：在线学习每个 minibatch 只有 ~20 帧，去相关后统计非效率
        g≈5-15 时常剩 2-7 帧，用默认的 10 帧门槛会让绝大多数 minibatch 被
        solve_stage_integrated 自身的逐 entry 门槛跳过、白白丢弃已经采到的真实
        数据。3 帧是能跑 MBAR 增广矩阵的下限，仍然不足以解出时返回 None，调用方
        按"这次没有可用更新"处理，不改动 f_k/Context。
        """
        if not self.tmbar_history:
            return None
        res = solve_stage_integrated(
            self.tmbar_history,
            self.kt,
            final_min_ess_ratio=min_ess_ratio,
            final_min_absolute_ess=min_absolute_ess,
            final_min_decorrelated_samples=min_decorrelated_samples,
            final_max_uncertainty_kJ_mol=max_uncertainty_kJ_mol,
            min_frames_per_window=min_frames_per_window,
            # [P1-19] self.tmbar_history 是容量固定的滑动 minibatch 历史，列表
            # 位置不是物理 λ 窗口——split-half 诊断的"window 0"在这条在线学习
            # 调用路径下没有实际含义，只会在滑窗塞满前对同一条最早 minibatch
            # 反复报出逐字节相同的告警。真正有意义的 split-half 审计在整段
            # 生产/最终分析的另外两个调用点上仍然照常运行。见
            # docs/experiments/P1-19_ONLINE_SLIDING_WINDOW_SPLITHALF_MISMATCH.md。
            skip_split_half_diagnostics=True,
        )
        if "error" in res:
            return None
        lambdas = list(res.get("lambdas") or [])
        f_k = list(res.get("f_k") or [])
        if sorted(lambdas) != list(range(self.n_states)) or len(f_k) != len(lambdas):
            return None
        f_by_lambda = {int(lam): float(f) for lam, f in zip(lambdas, f_k)}
        # [IBS_BIAS_PROTOCOL_VERSION=27] solve_stage_integrated 返回的 "f_k" 是
        # df_matrix[0, 1:] * kT，即 PyMBAR 的 f_i=-ln(Z_i)=beta*F_i 转成 kJ/mol
        # 后的物理 F_i（相差任意 gauge 常数）。IBS 混合权重满足
        # exp(beta*f_k) Z_k = exp[beta*(f_k-F_k)]，所以可直接使用同号的物理
        # F_k 并 mean-center；额外取负会把补偿方向翻转。瞬时占据过高时降低 f_k
        # 是下面 bounded occupancy feedback 的职责，不能据此反转物理 F_k。
        # v23 起这个绝对候选不写入 Context、只返回作诊断，但仍保持正确约定，
        # 避免未来调用者重新启用时继承旧的 v21 反号错误。
        f_new = np.array(
            [f_by_lambda[k] for k in range(self.n_states)], dtype=np.float64
        )
        f_new = f_new - float(np.mean(f_new))
        return f_new, res

    def _damped_tmbar_absolute_update(
        self,
        f_old: np.ndarray,
        f_tmbar: np.ndarray,
        damping: float = IBS_TMBAR_UPDATE_DAMPING,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Move toward the full-history absolute TMBAR solution.

        TMBAR already performs the self-consistent solve; repeatedly applying
        raw occupancy feedback to one minibatch would not add information.
        The damping controls statistical adaptation. This implements the
        literal convex iteration requested by the warm-up protocol; the
        pairwise trust radius is reserved for the no-TMBAR SGD fallback.
        """
        old = np.asarray(f_old, dtype=np.float64).ravel()
        target = np.asarray(f_tmbar, dtype=np.float64).ravel()
        if old.size != self.n_states or target.size != self.n_states:
            raise ValueError("f_old/f_tmbar 长度必须与 n_states 一致")
        if not np.all(np.isfinite(old)) or not np.all(np.isfinite(target)):
            raise ValueError("f_old/f_tmbar 必须全部有限")
        old = old - float(np.mean(old))
        target = target - float(np.mean(target))
        requested_damping = float(damping)
        if not 0.0 < requested_damping <= 1.0:
            raise ValueError("TMBAR damping 必须位于 (0, 1]")
        effective_damping = requested_damping
        delta_f = effective_damping * (target - old)
        delta_f -= float(np.mean(delta_f))
        pairwise_spread = (
            float(np.max(delta_f) - np.min(delta_f)) if delta_f.size else 0.0
        )
        f_new = old + delta_f
        f_new -= float(np.mean(f_new))
        diagnostics = {
            "method": "damped_absolute_tmbar_v9",
            "requested_damping": requested_damping,
            "effective_damping": float(effective_damping),
            "tmbar_candidate_f_kJ_mol": target.astype(float).tolist(),
            "tmbar_candidate_pairwise_span_kJ_mol": float(
                np.max(target) - np.min(target)
            ) if target.size else 0.0,
            "delta_f_kJ_mol": delta_f.astype(float).tolist(),
            "max_abs_delta_f_kJ_mol": float(np.max(np.abs(delta_f))) if delta_f.size else 0.0,
            "pairwise_delta_f_spread_kJ_mol": float(pairwise_spread),
            "pairwise_step_limit_kT": None,
            "pairwise_step_limit_kJ_mol": None,
        }
        return f_new, diagnostics

    def _bounded_log_occupancy_update(
        self,
        f_old: np.ndarray,
        mean_p: np.ndarray,
        max_abs_log_residual: float = 50.0,
        severe_max_pairwise_step_kT: float = 6.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply one sample-and-hold, pairwise-bounded all-state update.

        ``mean_p`` is the actually observed IBS responsibility under ``f_old``.
        The fixed point is ``K*<p_k> = 1``.  Updating by the *negative* log
        residual lowers every overrepresented state's f_k and raises every
        underrepresented state's f_k simultaneously.  This is an increment,
        not an absolute TMBAR vector; mean-centering only fixes the gauge after
        the increment has been applied.
        """
        f_old = np.asarray(f_old, dtype=np.float64).ravel()
        p = np.asarray(mean_p, dtype=np.float64).ravel()
        if f_old.size != self.n_states or p.size != self.n_states:
            raise ValueError("f_old/mean_p 长度必须与 n_states 一致")
        if not np.all(np.isfinite(f_old)) or not np.all(np.isfinite(p)):
            raise ValueError("f_old/mean_p 必须全部有限")
        if np.any(p < 0.0) or float(np.sum(p)) <= 0.0:
            raise ValueError("mean_p 必须是非负且总和为正")

        p = p / float(np.sum(p))
        p_safe = np.maximum(p, np.finfo(np.float64).tiny)
        raw_log_residual = np.log(float(self.n_states) * p_safe)
        clipped_log_residual = np.clip(
            raw_log_residual,
            -float(max_abs_log_residual),
            float(max_abs_log_residual),
        )
        # A common shift in all residuals is a pure f_k gauge change.  Remove
        # it before stepping so every update redistributes weight rather than
        # translating the whole vector.
        clipped_log_residual -= float(np.mean(clipped_log_residual))

        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=24] Denominator was /100 -- with a
        # per-window correction on the order of several kJ/mol needed
        # (log-residual ~1-2.5 -> delta_f ~kT*residual), that decayed eta
        # below a useful step size within the handful of updates it takes to
        # trip the candidate freeze gate, well before real convergence.
        # Slowed to /500 so eta stays close to full strength over the
        # window this decision actually needs to happen in.
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=25] v24's fix alone caused a NEW,
        # real GPU-observed failure mode: with eta no longer crushed, this
        # update applies delta_f to all K states *simultaneously*, each
        # sized off that state's own pre-update residual. For a
        # softmax-coupled mixture, lowering the (formerly) dominant state's
        # f_k while raising several underrepresented states' f_k in the same
        # step compounds the shift in relative log-odds well past the
        # single-state linear-response assumption this formula makes --
        # observed as delta_f staying pinned at 2-5 kJ/mol every update
        # while `dominant_k` flips between different states update to
        # update, never settling (classic control-loop overshoot, not slow
        # convergence). Warm-up updater v5 handles this with a tent-shaped
        # gain plus a bound on the pairwise Delta-f spread.  eta_penalty and
        # the update_index long-run decay remain independent safeguards.
        update_index = len(self.f_history) + 1
        residual_severity = float(np.max(np.abs(raw_log_residual)))
        if residual_severity >= IBS_UPDATE_ADAPTIVE_RESIDUAL_HIGH:
            gain_regime = "severe"
            adaptive_fraction = 1.0
            adaptive_relaxation = IBS_UPDATE_SEVERE_RELAXATION_FACTOR
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] severe 区分两档，不再从 4 kT 连续
            # 爬到 severe_max（那让 residual 十几的中等塌陷也吃 ~25 kJ/mol 的大步，
            # 表现为 dominant 每轮翻转、占据不收敛）。residual < COLLAPSE 保持中区
            # 顶端的 4 kT（≈10 kJ/mol，单次相对更新的安全上限）；只有 residual ≥
            # COLLAPSE 的真塌陷才放开到 caller 选的 severe_max——那种窗口用 4 kT/步
            # 需要十几二十步才建起所需 f_k 谱宽，慢到会吃光 max_bias_updates 预算。
            if residual_severity >= IBS_UPDATE_ADAPTIVE_RESIDUAL_COLLAPSE:
                pairwise_step_limit_kT = float(severe_max_pairwise_step_kT)
            else:
                pairwise_step_limit_kT = 4.0
        elif residual_severity >= IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW:
            gain_regime = "middle"
            adaptive_fraction = float(
                (residual_severity - IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW)
                / (
                    IBS_UPDATE_ADAPTIVE_RESIDUAL_HIGH
                    - IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW
                )
            )
            # At the edge of the near-flat basin gain is highest, because the
            # response is observable but not saturated.  It falls toward the
            # severe value as collapse increases. The pairwise radius grows
            # from 2 to 4 kT across this recoverable middle regime.
            adaptive_relaxation = float(
                IBS_UPDATE_MIDDLE_RELAXATION_FACTOR
                + adaptive_fraction
                * (
                    IBS_UPDATE_SEVERE_RELAXATION_FACTOR
                    - IBS_UPDATE_MIDDLE_RELAXATION_FACTOR
                )
            )
            pairwise_step_limit_kT = float(2.0 + 2.0 * adaptive_fraction)
        else:
            gain_regime = "near_flat"
            adaptive_fraction = float(
                residual_severity / IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW
            )
            # Do not chase finite-block argmax noise at the fixed point.
            adaptive_relaxation = float(
                IBS_UPDATE_NEAR_FLAT_RELAXATION_FACTOR
                + adaptive_fraction
                * (
                    IBS_UPDATE_MIDDLE_RELAXATION_FACTOR
                    - IBS_UPDATE_NEAR_FLAT_RELAXATION_FACTOR
                )
            )
            pairwise_step_limit_kT = float(0.5 + 1.5 * adaptive_fraction)
        eta = (
            adaptive_relaxation
            * float(self.eta_penalty)
            / (1.0 + float(update_index) / 500.0)
        )
        delta_f = -eta * float(self.kt) * clipped_log_residual

        # What changes softmax odds is delta_f[i]-delta_f[j], not an absolute
        # component relative to an arbitrary mean-zero gauge.  Bound that
        # physically meaningful pairwise spread and preserve the direction by
        # uniform rescaling.  In a one-dominant five-state collapse, a 3-kT
        # The caller selects a severe ceiling only after checking dominant
        # persistence. This is still a relative-odds bound, not an arbitrary
        # gauge-dependent per-component radius.
        pairwise_step_limit_kj = pairwise_step_limit_kT * float(self.kt)
        pairwise_delta_spread = (
            float(np.max(delta_f) - np.min(delta_f)) if delta_f.size else 0.0
        )
        if pairwise_delta_spread > pairwise_step_limit_kj > 0.0:
            delta_f *= pairwise_step_limit_kj / pairwise_delta_spread
            pairwise_delta_spread = float(np.max(delta_f) - np.min(delta_f))

        f_new = f_old + delta_f
        f_new -= float(np.mean(f_new))
        diagnostics = {
            "method": "bounded_log_occupancy_fallback_v9",
            "eta": float(eta),
            "eta_penalty": float(self.eta_penalty),
            "residual_severity": float(residual_severity),
            "adaptive_relaxation": float(adaptive_relaxation),
            "adaptive_fraction": float(adaptive_fraction),
            "gain_regime": gain_regime,
            "raw_log_residual": raw_log_residual.astype(float).tolist(),
            "clipped_centered_log_residual": clipped_log_residual.astype(float).tolist(),
            "delta_f_kJ_mol": delta_f.astype(float).tolist(),
            "max_abs_delta_f_kJ_mol": float(np.max(np.abs(delta_f))) if delta_f.size else 0.0,
            "pairwise_delta_f_spread_kJ_mol": float(pairwise_delta_spread),
            "pairwise_step_limit_kT": float(pairwise_step_limit_kT),
            "pairwise_step_limit_kJ_mol": float(pairwise_step_limit_kj),
        }
        return f_new, diagnostics

    @staticmethod
    def _apply_pairwise_cap(
        f_old: np.ndarray,
        f_candidate: np.ndarray,
        cap_kt: float,
        kt: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Hard-cap the pairwise (max-min) spread of (f_candidate - f_old) to
        ``cap_kt*kt`` kJ/mol, preserving direction via uniform rescaling, then
        mean-center. Pure function -- no sampler state.

        [Candidate-first, Validate-or-Learn v1] Extracted from the hard-cap
        block previously inlined in ``update_weights()`` (bit-identical math)
        so both the trusted-absolute-TMBAR path there and the VALIDATE-
        failure damped-correction retry path in ``run_all_windows`` share one
        implementation instead of two copies that can silently drift apart.
        """
        old = np.asarray(f_old, dtype=np.float64).ravel()
        candidate = np.asarray(f_candidate, dtype=np.float64).ravel()
        delta = candidate - old
        delta -= float(np.mean(delta)) if delta.size else 0.0
        spread = float(np.max(delta) - np.min(delta)) if delta.size else 0.0
        cap_kj = float(cap_kt) * float(kt)
        capped = False
        if spread > cap_kj > 0.0:
            delta = delta * (cap_kj / spread)
            spread = float(np.max(delta) - np.min(delta))
            capped = True
        f_new = old + delta
        f_new -= float(np.mean(f_new))
        return f_new, {
            "hard_pairwise_cap_applied": capped,
            "hard_pairwise_cap_kJ_mol": cap_kj,
            "delta_f_kJ_mol": delta.astype(float).tolist(),
            "pairwise_delta_f_spread_kJ_mol": float(spread),
        }

    def update_weights(
        self,
        min_buffer_size: int = 20,
        candidate_min_ess_ratio: float = 0.05,
        candidate_min_absolute_ess: float = 1.0,
        # [v23 numerical follow-up #3] Was 5 -- mathematically unreachable
        # given `_solve_tmbar_and_recenter`'s own `min_frames_per_window=3`
        # (documented there as the actual floor for running the augmented
        # MBAR matrix at all, precisely because decorrelation commonly
        # leaves only 2-7 frames per ~20-frame minibatch). Any minibatch with
        # 3 or 4 decorrelated frames is deliberately let into local_results
        # by that floor, which then caps the aggregate min_decorrelated_
        # samples at 3 or 4 for as long as such a minibatch exists in
        # tmbar_history -- requiring >=5 here made this candidate criterion
        # permanently unsatisfiable in exactly the regime it's meant to
        # evaluate, same failure shape as the min_absolute_ess 2.5->1.0 fix
        # above. Lowered to 3 to match the actual floor; final_*/frozen-
        # validation gates are untouched.
        candidate_min_decorrelated_samples: int = 3,
        candidate_max_uncertainty_kJ_mol: float = 5.0,
    ) -> Optional[np.ndarray]:
        """
        在线 IBS 权重更新。v9 使用全部累计 minibatch 的 TMBAR 绝对候选，
        按固定 0.20 阻尼更新 f_k。``mean_p_batch``、EMA 和 dominant 仅供诊断；
        只有 TMBAR 暂不可解时才执行一次 bounded-SGD 兜底（pairwise 半径 4 kT，
        仅 raw_residual≥70 的真塌陷放开到 10 kT）。
        candidate_* 阈值同时写入 tmbar_update 诊断，供 learning 候选门读取。
        """
        if len(self.energy_buffer) < min_buffer_size:
            return None

        u_mk = np.array(self.energy_buffer) # (M, K)
        valid_mask = ~np.isnan(u_mk).any(axis=1)
        if np.sum(valid_mask) < min_buffer_size:
            return None

        u_mk = u_mk[valid_mask]
        M, K = u_mk.shape
        
        # 1. 获取当前 f_old
        f_old = np.array([self.context.getParameter(f"{self.prefix}_f_{k}") for k in range(K)])
        
        # 2. 计算瞬时概率 p_ik
        # 注意：u_mk 已经是相对于 State 0 偏移后的能量
        # 公式: p_ik ∝ exp(-beta * (u_mk[i,k] - f_k))
        # 由于 u_mk 是相对值，f_k 也应当被解释为相对自由能估计
        
        beta_u = -self.beta * u_mk  # (M, K)
        beta_f = self.beta * f_old  # (K,)
        
        logits = beta_f[None, :] + beta_u  # (M, K)
        
        # Numerical stability: log-sum-exp
        max_logits = np.max(logits, axis=1, keepdims=True)
        log_denom = max_logits.squeeze() + np.log(np.sum(np.exp(logits - max_logits), axis=1))
        
        log_p = logits - log_denom[:, None]
        p_ik = np.exp(log_p)  # (M, K)
        
        # 3. 计算批次平均概率
        mean_p_batch = np.mean(p_ik, axis=0)  # (K,)

        adjacent_delta_u = np.diff(u_mk, axis=1)
        adjacent_records = []
        for edge_idx in range(max(K - 1, 0)):
            edge = np.asarray(adjacent_delta_u[:, edge_idx], dtype=float)
            adjacent_records.append({
                "edge": [int(edge_idx), int(edge_idx + 1)],
                "mean_delta_u_kJ_mol": float(np.mean(edge)),
                "std_delta_u_kJ_mol": float(np.std(edge, ddof=1)) if edge.size > 1 else 0.0,
                "std_reduced_delta_u": (
                    float(self.beta * np.std(edge, ddof=1)) if edge.size > 1 else 0.0
                ),
                "rms_delta_u_kJ_mol": float(np.sqrt(np.mean(edge * edge))),
                "p05_delta_u_kJ_mol": float(np.percentile(edge, 5.0)),
                "p50_delta_u_kJ_mol": float(np.percentile(edge, 50.0)),
                "p95_delta_u_kJ_mol": float(np.percentile(edge, 95.0)),
            })
        self.last_update_diagnostics = {
            "batch_size": int(M),
            "mean_p_batch": mean_p_batch.astype(float).tolist(),
            "lse_balance": ibs_lse_balance_diagnostics(mean_p_batch),
            "mean_u_kJ_mol": np.mean(u_mk, axis=0).astype(float).tolist(),
            "adjacent_delta_u": adjacent_records,
        }
        
        # 4. EMA 更新（仅供原始占据趋势诊断）。
        if self.ema_mean_p is None:
            self.ema_mean_p = mean_p_batch.copy()
        else:
            self.ema_mean_p = self.gamma * self.ema_mean_p + (1.0 - self.gamma) * mean_p_batch

        # 5. 论文 eq. 15 TMBAR：把这批 minibatch 打包进持久历史，用累计
        # 至今的全部时变采样分布自洽求解绝对物理自由能。v9 起 f_k 更新只朝
        # 这个全历史绝对候选阻尼移动；当前 raw batch 占据仅作诊断，
        # 不再充当 SGD 更新方向。
        self._append_tmbar_batch_from_buffer()
        tmbar_result = self._solve_tmbar_and_recenter(
            min_ess_ratio=candidate_min_ess_ratio,
            min_absolute_ess=candidate_min_absolute_ess,
            min_decorrelated_samples=candidate_min_decorrelated_samples,
            max_uncertainty_kJ_mol=candidate_max_uncertainty_kJ_mol,
        )
        if tmbar_result is None:
            tmbar_absolute_candidate = None
            self.last_update_diagnostics["tmbar_update"] = {
                "available": False,
                "n_tmbar_entries": len(self.tmbar_history),
                "converged": False,
            }
        else:
            tmbar_absolute_candidate, tmbar_res = tmbar_result
            self.last_update_diagnostics["tmbar_update"] = {
                "available": True,
                "n_tmbar_entries": len(self.tmbar_history),
                # 🔑 [2026-09-15] solve_stage_integrated 的 `converged` 键已删除；
                # 这里改读 `analysis_status`（"路径完整 + 数值有限 + 结构自洽"，
                # 不含任何阈值）。它跟以前那个 legacy flag 一样只是诊断：在线
                # readiness 在下面由真正的阻尼 TMBAR 不动点步赋值。
                "analysis_status": tmbar_res.get("analysis_status"),
                "analysis_incomplete_reasons": tmbar_res.get(
                    "analysis_incomplete_reasons"
                ),
                # [ESS_GATE_PROTOCOL_VERSION=2] raw_* 是这条路径真正用来判 trust 的量
                # （见下面 tmbar_candidate_trusted 处的说明）；mixture 量对 tmbar_history
                # 恒为 None，一并落盘只为让诊断自解释。
                "min_overlap": tmbar_res.get("raw_min_overlap"),
                "min_absolute_ess": tmbar_res.get("raw_min_absolute_ess"),
                "mixture_min_overlap": tmbar_res.get("min_overlap"),
                "min_decorrelated_samples": tmbar_res.get("min_decorrelated_samples"),
                "max_endpoint_uncertainty_kJ_mol": tmbar_res.get("max_endpoint_uncertainty_kJ_mol"),
            }
        self.last_update_diagnostics["adjacent_delta_u_is_convergence_gate"] = False

        dominant_k = int(np.argmax(mean_p_batch))
        raw_log_residual = np.log(
            float(K) * np.maximum(mean_p_batch, np.finfo(np.float64).tiny)
        )
        residual_severity = float(np.max(np.abs(raw_log_residual)))
        total_tmbar_frames = int(sum(
            np.asarray(entry.get("u_kn", np.empty((0, 0)))).shape[1]
            for entry in self.tmbar_history
        ))
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 只有累计 TMBAR 解质量同时可信时才应用
        # 绝对候选；否则退回受限占据反馈先恢复覆盖，避免"低重叠 TMBAR 给出错误
        # 大步 → 占据塌缩 → TMBAR 更不可靠"的正反馈。coverage_ess 用当前 batch 占据
        # 算（mean_p_batch 已归一，sum=1）。这些只门控"用哪个控制器"，不是窗口
        # 收敛判据——收敛仍是 run_all_windows 里的 local-MBAR loose gate。
        _sum_sq_p = float(np.sum(np.square(mean_p_batch)))
        coverage_ess_batch = float(1.0 / _sum_sq_p) if _sum_sq_p > 0.0 else 0.0
        tmbar_candidate_trusted = False
        if tmbar_absolute_candidate is not None:
            # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] 这里刻意读 raw_* 而不是 min_overlap/
            # min_absolute_ess。两个消费者要的不是同一件事：
            #   - 阶段收敛门问"混合分布对物理态覆盖够不够"→ 用扣掉共模因子的
            #     mixture 量（共模项在它真正报出去的物理态↔物理态 ΔF 里会抵消）。
            #   - 这里问"从这批原始重要性权重解出来的绝对 f_k 向量敢不敢直接用"→
            #     权重退化本身就是不敢用的理由，raw 单参考 ESS 才是正确的悲观度量。
            # 另外 tmbar_history 的 entry 天生没有 f_k 字段（每条 entry 是在不同
            # f_k 下采的，单一 f_k 无法代表整段历史，这正是 TMBAR 存在的理由），
            # mixture 量对这个调用方恒为 None——读它会让 trust 门永久为 False，
            # 把 warmup 永久钉死在受限占据反馈上。
            _q_overlap = tmbar_res.get("raw_min_overlap")
            _q_abs_ess = tmbar_res.get("raw_min_absolute_ess")
            _q_decorr = tmbar_res.get("min_decorrelated_samples")
            _q_uncert = tmbar_res.get("max_endpoint_uncertainty_kJ_mol")
            tmbar_candidate_trusted = bool(
                _q_overlap is not None
                and _q_overlap >= IBS_TMBAR_TRUST_MIN_OVERLAP
                and _q_abs_ess is not None
                and _q_abs_ess >= IBS_TMBAR_TRUST_MIN_ABSOLUTE_ESS
                and _q_decorr is not None
                and _q_decorr >= IBS_TMBAR_TRUST_MIN_DECORRELATED_SAMPLES
                and _q_uncert is not None
                and np.isfinite(_q_uncert)
                and _q_uncert <= IBS_TMBAR_TRUST_MAX_UNCERTAINTY_KJ_MOL
                and coverage_ess_batch
                >= IBS_TMBAR_TRUST_MIN_COVERAGE_ESS_FRACTION * float(K)
            )
        if tmbar_candidate_trusted:
            f_new, weight_update_diag = self._damped_tmbar_absolute_update(
                f_old,
                tmbar_absolute_candidate,
            )
        else:
            # TMBAR 候选不可信（低重叠/低 ESS/去相关不足/不确定度大/覆盖差）或
            # 根本不可解：用受限占据反馈先把覆盖拉回来（Δf_k=-η·kT·ln(K·p_k)，
            # 自带自适应 pairwise 上限），不让一步错误大步打崩占据。
            f_new, weight_update_diag = self._bounded_log_occupancy_update(
                f_old,
                mean_p_batch,
                severe_max_pairwise_step_kT=(
                    IBS_TMBAR_FALLBACK_SGD_PAIRWISE_STEP_KT
                ),
            )
            weight_update_diag["method"] = (
                "bounded_occupancy_tmbar_untrusted_v9"
                if tmbar_absolute_candidate is not None
                else "bounded_sgd_fallback_v9"
            )
        weight_update_diag["tmbar_candidate_trusted"] = bool(tmbar_candidate_trusted)
        weight_update_diag["coverage_ess_batch"] = float(coverage_ess_batch)
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 硬 pairwise cap 只加在【可信绝对 TMBAR】
        # 路径上：绝对候选即使可信、估计略偏时一大步也危险，故 cap 到
        # IBS_MAX_APPLIED_PAIRWISE_STEP_KT(2 kT)。而 bounded occupancy 反馈是自我纠偏
        # 的比例控制器（Δf=-η·kT·ln(K·p)），且严重塌陷时本就需要大步纠偏——它自带
        # 的自适应 pairwise 上限（近平坦 0.5→中区 2→4→中等塌陷 4→真塌陷 severe_max）已
        # 足够安全，这里不再额外硬 cap 到 2 kT，否则 raw_residual≈150 的严重窗口会被
        # 卡在 2 kT/步、要 ~75 步才建起所需 f_k 谱宽（塌陷的正反馈风险来自已被 trust-
        # gate 关掉的绝对 TMBAR，不来自 bounded）。诊断一律刷新成 cap 之后的真实步长。
        if tmbar_candidate_trusted:
            # [Candidate-first, Validate-or-Learn v1] 硬 cap 逻辑已提取为
            # 独立纯函数 _apply_pairwise_cap（跟 VALIDATE 失败重试路径共用
            # 同一实现），行为与之前内联版本逐字节一致。
            f_new, _cap_diag = self._apply_pairwise_cap(
                f_old, f_new, IBS_MAX_APPLIED_PAIRWISE_STEP_KT, self.kt
            )
            weight_update_diag["hard_pairwise_cap_applied"] = _cap_diag["hard_pairwise_cap_applied"]
            weight_update_diag["hard_pairwise_cap_kJ_mol"] = _cap_diag["hard_pairwise_cap_kJ_mol"]
            _applied_delta = np.asarray(_cap_diag["delta_f_kJ_mol"], dtype=np.float64)
            _applied_spread = float(_cap_diag["pairwise_delta_f_spread_kJ_mol"])
        else:
            # bounded 路径：不额外硬 cap，用其自带自适应上限（见 _bounded_log_
            # occupancy_update）。诊断记录未施加外部硬 cap。
            _applied_delta = (
                np.asarray(f_new, dtype=np.float64)
                - np.asarray(f_old, dtype=np.float64)
            )
            _applied_delta -= float(np.mean(_applied_delta))
            _applied_spread = (
                float(np.max(_applied_delta) - np.min(_applied_delta))
                if _applied_delta.size else 0.0
            )
            weight_update_diag["hard_pairwise_cap_applied"] = False
            weight_update_diag["hard_pairwise_cap_kJ_mol"] = None
        weight_update_diag["delta_f_kJ_mol"] = _applied_delta.astype(float).tolist()
        weight_update_diag["max_abs_delta_f_kJ_mol"] = (
            float(np.max(np.abs(_applied_delta))) if _applied_delta.size else 0.0
        )
        weight_update_diag["pairwise_delta_f_spread_kJ_mol"] = float(_applied_spread)
        weight_update_diag["residual_severity"] = float(residual_severity)
        weight_update_diag["total_tmbar_frames"] = int(total_tmbar_frames)
        applied_pairwise_step = float(
            weight_update_diag.get("pairwise_delta_f_spread_kJ_mol", float("inf"))
        )
        self_consistency_limit_kj = float(
            IBS_TMBAR_FREEZE_MAX_APPLIED_PAIRWISE_STEP_KT * self.kt
        )
        tmbar_self_consistent = bool(
            tmbar_absolute_candidate is not None
            and np.isfinite(applied_pairwise_step)
            and applied_pairwise_step <= self_consistency_limit_kj
        )
        self.last_update_diagnostics["tmbar_update"].update({
            "converged": tmbar_self_consistent,
            "online_convergence_method": "damped_absolute_step_pairwise",
            "applied_pairwise_step_kJ_mol": applied_pairwise_step,
            "applied_pairwise_step_threshold_kJ_mol": self_consistency_limit_kj,
            "total_tmbar_frames": int(total_tmbar_frames),
        })
        self.last_update_diagnostics["weight_update"] = weight_update_diag
        # [P1-19] `effective_damping` 只在走了可信 TMBAR 绝对更新
        # （_damped_tmbar_absolute_update）时才被写进 weight_update_diag；
        # occupancy/SGD fallback 路径没有"阻尼系数"这个概念。此前用
        # `.get(..., 0.0)` 兜底会在 fallback 路径把 alpha 打印成 0.000，
        # 看起来像"阻尼到零但 delta_f 仍非零"，实为字段不存在，不是阻尼系数
        # 真的是 0。fallback 时改打印 "n/a" 并显式带上 method 是否可信。
        _alpha = weight_update_diag.get("effective_damping")
        _alpha_str = f"{float(_alpha):.3f}" if _alpha is not None else "n/a(fallback)"
        print(
            f"    [IBS TMBAR 自洽权重更新 v9] dominant诊断=state{dominant_k} "
            f"p={float(mean_p_batch[dominant_k]):.6f}, "
            f"delta_f={float(weight_update_diag['delta_f_kJ_mol'][dominant_k]):+.3f} kJ/mol, "
            f"max|delta_f|={float(weight_update_diag['max_abs_delta_f_kJ_mol']):.3f} "
            f"(pairwise={float(weight_update_diag['pairwise_delta_f_spread_kJ_mol']):.3f} kJ/mol, "
            f"method={weight_update_diag['method']}, trusted={tmbar_candidate_trusted}, "
            f"alpha={_alpha_str}, "
            f"raw_residual={residual_severity:.3f}, "
            f"batch={M}, sliding_window_frames={total_tmbar_frames}, "
            # [P1-19] 这个字段只是"本次实际应用的 pairwise 步长 ≤ 自洽阈值"，
            # 与 method/trusted 是否走了可信 TMBAR 绝对更新正交——method=
            # bounded_occupancy...untrusted 时它仍可能是 True（fallback 自己的
            # 步长恰好也小），不代表 TMBAR 候选本身可信/被采用，改名避免
            # 读成"TMBAR 自洽"。
            f"applied_step_within_selfconsistency_limit={tmbar_self_consistent}, "
            f"analysis_status={self.last_update_diagnostics['tmbar_update'].get('analysis_status')})"
        )
        # Dominant identity remains a diagnostic breadcrumb only. It does not
        # reset, brake, accelerate, or otherwise alter the TMBAR update.
        self._last_dominant_k = dominant_k

        # 6. 应用更新
        for k in range(K):
            self.context.setParameter(f"{self.prefix}_f_{k}", float(f_new[k]))

        self.energy_buffer = []
        self.f_history.append(f_new.copy())
        return f_new

    def apply_learning_rate_penalty(self, factor: float = 0.85, floor: float = 0.25) -> float:
        """Damp (default 0.85x) the learning-rate multiplier after a frozen-
        validation failure, so the next learning attempt takes more
        conservative steps instead of oscillating with the same large step
        size that just failed. Floored to avoid learning stalling out
        entirely after repeated failures.

        [IBS_BIAS_PROTOCOL_VERSION=24] factor was 0.5 and floor was 0.05:
        after just 5 real failures (0.5**5 = 0.03125 < floor), eta_penalty
        bottomed out at 1/20th strength and stayed there across every future
        process restart (it's persisted -- see save/load_ibs_state), because
        nothing ever raises it back up. That silently capped every future
        learning attempt to steps far too small to close a multi-kJ/mol
        occupancy gap, guaranteeing repeated freeze/validate failures that
        each burn the full step budget for no real progress. 0.85/0.25 keep
        the same "get more conservative after a real failure" intent without
        collapsing to a step size that can no longer converge at all.
        """
        self.eta_penalty = max(float(floor), float(self.eta_penalty) * float(factor))
        return self.eta_penalty

    def evaluate_frozen_batch_probability(self, min_buffer_size: int = 20) -> Optional[np.ndarray]:
        """Measure this batch's mean occupation probability under the CURRENT
        f_k, without applying any update.

        ``update_weights()`` computes its probability estimate from ``f_old``
        and then immediately overwrites the context with ``f_new`` in the same
        call -- so the f_k that actually ends up frozen for production has
        never itself been the subject of a passing check; every "consecutive
        pass" was evidence about a different, already-discarded f_k. This
        method is the frozen-validation counterpart: same log-sum-exp
        probability math, same EMA update, but it reads f_k and never writes
        it, so repeated calls against an unchanged f_k are genuine repeated
        evidence about that exact f_k.
        """
        if len(self.energy_buffer) < min_buffer_size:
            return None
        u_mk = np.array(self.energy_buffer)
        valid_mask = ~np.isnan(u_mk).any(axis=1)
        if np.sum(valid_mask) < min_buffer_size:
            return None
        u_mk = u_mk[valid_mask]
        M, K = u_mk.shape
        self.last_frozen_batch_size = int(M)

        f_k = np.array([self.context.getParameter(f"{self.prefix}_f_{k}") for k in range(K)])
        beta_u = -self.beta * u_mk
        beta_f = self.beta * f_k
        logits = beta_f[None, :] + beta_u
        max_logits = np.max(logits, axis=1, keepdims=True)
        log_denom = max_logits.squeeze() + np.log(np.sum(np.exp(logits - max_logits), axis=1))
        log_p = logits - log_denom[:, None]
        p_ik = np.exp(log_p)
        mean_p_batch = np.mean(p_ik, axis=0)

        if self.ema_mean_p is None:
            self.ema_mean_p = mean_p_batch.copy()
        else:
            self.ema_mean_p = self.gamma * self.ema_mean_p + (1.0 - self.gamma) * mean_p_batch

        self.energy_buffer = []
        return mean_p_batch
# ================= ibs_engine.py -> IBSSampler 类 =================
    def save_ibs_state(
        self,
        filepath: str,
        lambdas_coul=None,
        lambdas_vdw=None,
        stage_type: Optional[str] = None,
    ):
        """同步保存 IBS 状态，使用原子替换避免损坏。

        lambdas_coul/lambdas_vdw：本窗口这次真正对应的 λ 值（跟 convergence.json
        里存的是同一份东西）。之前这里没存——resume 只按 n_states 数量判断"能不能
        用"，λ 路径被自动加密/重新划分窗口后，state 数量凑巧相同但 f_k[k] 早就不
        对应同一个 λ 了，却仍会被当成有效热启动注入，这是错的。
        """
        f_current = [self.context.getParameter(f"{self.prefix}_f_{k}") for k in range(self.n_states)]
        state = {
            "n_states": int(self.n_states),
            "prefix": self.prefix,
            "f_k": f_current,
            "t": len(self.f_history),
            # [IBS_BIAS_PROTOCOL_VERSION=23] 有界增量的学习率惩罚属于续算状态；
            # 冻结验证失败后会降低，resume 时必须保持，不能静默重置为 1.0。
            "eta_penalty": float(self.eta_penalty),
            "e_offset": self.e_offset,
            "f_history_kj_mol": [
                np.asarray(values, dtype=float).tolist()
                for values in self.f_history
            ],
            # [IBS_BIAS_PROTOCOL_VERSION=19] 论文 eq. 15 TMBAR 的持久 minibatch
            # 历史（取代 v18 及更早版本的 ta_relative_q_sum/ta_iteration_count），
            # 是恢复在线学习进度必须的一部分——跟旧字段一样，load 时要在注入
            # 任何 f_k 之前完整校验。
            "tmbar_history": [
                {
                    "u_kn": np.asarray(entry["u_kn"], dtype=float).tolist(),
                    "bias_energies": np.asarray(
                        entry["bias_energies"], dtype=float
                    ).tolist(),
                    "base_energies": np.asarray(
                        entry["base_energies"], dtype=float
                    ).tolist(),
                    "lambda_indices": [int(x) for x in entry["lambda_indices"]],
                    "sampled_distribution_row": int(
                        entry.get("sampled_distribution_row", 0)
                    ),
                }
                for entry in self.tmbar_history
            ],
            "tmbar_history_dropped_entries": int(
                self.tmbar_history_dropped_entries
            ),
            "status": "running",
            # 🔑 见 IBS_BIAS_PROTOCOL_VERSION 定义处：只有这里存的 bias_converged=True
            # 才允许 load_ibs_state 之后直接跳过 Warmup 进生产；否则续传必须继续走
            # 严格收敛判据，不能假设旧状态已经收敛。
            "bias_converged": bool(self.bias_converged),
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12] bias_status/frozen_f_k_pending：
            # 见 IBSSampler.__init__ 里的注释。resume 只有在 bias_status ==
            # "calibrated_pending_validation" 且 frozen_f_k_pending 非空时才会
            # 跳过 [learning]、直接带着这份冻结 f_k 回到 freeze_burn_in。
            "bias_status": str(self.bias_status),
            "frozen_f_k_pending": (
                [float(x) for x in self.frozen_f_k_pending]
                if self.frozen_f_k_pending is not None else None
            ),
            # 🔑 见 IBSSampler.__init__ 里的注释——这份冻结 f_k 累计已经花在冻结
            # 验证上的步数，跨越同一份校准 f_k 的多次 resume/阶梯升级累加。
            "frozen_validation_cumulative_steps": int(self.frozen_validation_cumulative_steps),
            # 🔑 [2026-09-11] frozen_validation_indeterminate 下已累计的固定-f_k
            # 验证批数；resume 据此接着累计，不重新领额度。见 __init__ 注释。
            "frozen_validation_batches_done": int(
                getattr(self, "frozen_validation_batches_done", 0)
            ),
            # 🔑 [2026-09-11] 见 new_warmup_budget_ledger 的注释。
            "f_k_evidence_status": str(
                getattr(self, "f_k_evidence_status", F_K_EVIDENCE_NONE)
            ),
            "warmup_budget_ledger": (
                dict(self.warmup_budget_ledger)
                if getattr(self, "warmup_budget_ledger", None) else None
            ),
            "frozen_candidate_fingerprint": getattr(
                self, "frozen_candidate_fingerprint", None
            ),
            "ibs_bias_protocol_version": IBS_BIAS_PROTOCOL_VERSION,
            "warmup_update_protocol_version": IBS_WARMUP_UPDATE_PROTOCOL_VERSION,
            # 🔑 [non_mutating_v1] 记录产出这份 f_k 状态的采样修复策略。旧的变异
            # 策略可能就地重校准过 f_k（不同参考系）；load 时据此 fail-closed。
            "sampling_repair_policy": getattr(self, "sampling_repair_policy", None),
            "sampling_score_sha256": getattr(self, "sampling_score_sha256", None),
            # 🔑 [2026-09-11] 采样身份（λ / Hamiltonian / box / 口径版本…）。
            # 老板定案：「"已收敛窗口不降级"应限定为**同一个完整 ensemble
            # fingerprint**；如果 λ、Hamiltonian、box 或规范版本变化，旧 PASS 不能
            # 继续粘住。」实测这个洞真发生过：能量缓存因 stage_protocol_key 不符被
            # 拒、整窗重采，而这份状态里的 bias_converged=True 照样粘了过去，于是
            # 窗口跳过 learning 只做"只读复验"—— 复验的是一个**不同系综**的旧结论。
            "stage_protocol_key": getattr(self, "stage_protocol_key", None),
            # co-ion 身份同属"这是不是同一个 Hamiltonian"那一组证据（MEM-00c 冻结的
            # 那条）。`convergence.json` 一直记着它，state 这边却从来没写过 ——
            # 对账时只有一半，等于少一个信源。
            "coion_identity": getattr(self, "coion_identity", None),
            "residual_sampling_protocol_version": (
                IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION
                if getattr(self, "sampling_score_sha256", None) is not None
                else None
            ),
            "lambdas_coul": [float(x) for x in lambdas_coul] if lambdas_coul is not None else None,
            "lambdas_vdw": [float(x) for x in lambdas_vdw] if lambdas_vdw is not None else None,
            # [Candidate-first, Validate-or-Learn v1] 纯附加元数据，绝不参与
            # load_ibs_state 的任何 fail-closed/版本判定。seed_source 只是
            # "这份 f_k 是怎么来的"的事后可读标签；validation_attempts 只数
            # 真正跑完一次 local-MBAR gate 评估的次数（不含 burn-in/早退）；
            # last_failure_reason 是最近一次未收敛/失败的具体原因，供事后
            # 排障，不驱动任何 resume 分支。learning_updates 是 "t" 的只读
            # 别名，只在保存时写一次，绝不在 load 时独立读回——避免出现两个
            # 可能漂移的"学习次数"来源。
            "seed_source": getattr(self, "seed_source", None),
            "validation_attempts": int(getattr(self, "validation_attempts", 0) or 0),
            "last_failure_reason": getattr(self, "last_failure_reason", None),
            "learning_updates": len(self.f_history),
            # 🔑 [2026-08-27，见 docs/experiments/EXP-030_FROZEN_SNAPSHOT_TIMING_BUG_2026-08-26.md
            # 补验收] 只在 run_all_windows 真正进入生产采样那一刻被设置一次
            # （见该处注释），之后永不再被赋值——所以每次 save_ibs_state 落盘的
            # 都是同一份"生产开始时"的快照，不会被生产期间的任何操作污染。
            # None 表示这个窗口还没跑到生产采样阶段（纯校准/学习期间的存档）。
            "production_entry_f_k": (
                [float(x) for x in getattr(self, "production_entry_f_k")]
                if getattr(self, "production_entry_f_k", None) is not None else None
            ),
        }
        if stage_type == "vdw":
            state["vdw_nonbonded_protocol_version"] = VDW_NONBONDED_PROTOCOL_VERSION
        if getattr(self, "sampling_score_sha256", None) is not None:
            state["residual_sampling"] = {
                "feature": getattr(
                    self, "residual_feature_name", "Outer-Lambda Local Residual for IBS"
                ),
                "em_policy": getattr(self, "residual_em_policy", "no_residual_twin"),
                "plugin_identity": getattr(self, "residual_plugin_identity", None),
            }
        _atomic_write_json(filepath, state)

    def load_ibs_state(
        self,
        filepath: str,
        lambdas_coul=None,
        lambdas_vdw=None,
        stage_type: Optional[str] = None,
    ) -> bool:
        """反序列化并注入 IBS 状态，恢复历史记忆。

        🔑 n_states/prefix/协议版本/λ 内容任何一项对不上，一律完全拒绝这份旧
        状态。只有这些基础身份字段都匹配、即 state 确实声称属于当前协议时，才
        检查更严格的 sampling_repair_policy：当前 policy 为 non_mutating_v1 时，
        当前协议的缓存若缺字段或来自其它 policy，不允许静默 return False 后在原
        目录重采，必须在注入任何 f_k 之前抛
        ExistingEnsembleRequiresRescueAudit，保留原 ensemble 交 rescue audit。
        旧协议 state 则走正常的版本失配路径安全失效，绝不注入 f_k。
        """
        import json, os
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)
            if state.get("n_states") != self.n_states:
                print(
                    f"  [WARN] IBS 状态与当前窗口不兼容 "
                    f"(cache n_states={state.get('n_states')}, current={self.n_states})，忽略旧状态"
                )
                return False
            if state.get("prefix") not in (None, self.prefix):
                print(
                    f"  [WARN] IBS 状态 prefix 不兼容 "
                    f"(cache={state.get('prefix')}, current={self.prefix})，忽略旧状态"
                )
                return False
            cached_protocol_version = state.get("ibs_bias_protocol_version")
            if not _ibs_bias_protocol_version_is_cache_compatible(
                cached_protocol_version
            ):
                print(
                    f"  [WARN] IBS 状态协议版本不匹配 (cache={cached_protocol_version!r}, "
                    f"兼容版本={sorted(IBS_BIAS_CACHE_COMPATIBLE_PROTOCOL_VERSIONS)})，"
                    "完全忽略旧状态（不作为热启动），"
                    "从 f_k=0 重新开始"
                )
                return False
            cached_warmup_update_version = state.get(
                "warmup_update_protocol_version"
            )
            if (
                not bool(state.get("bias_converged", False))
                and cached_warmup_update_version
                != IBS_WARMUP_UPDATE_PROTOCOL_VERSION
            ):
                print(
                    "  [WARN] 未完成的 IBS 预热状态使用旧权重控制器 "
                    f"(cache={cached_warmup_update_version!r}, "
                    f"current={IBS_WARMUP_UPDATE_PROTOCOL_VERSION})，"
                    "拒绝把旧控制器的中间 f_k 注入新自适应学习；从当前协议重新预热"
                )
                return False
            cached_lc = state.get("lambdas_coul")
            cached_lv = state.get("lambdas_vdw")
            lambdas_match = (
                lambdas_coul is not None
                and lambdas_vdw is not None
                and cached_lc is not None
                and cached_lv is not None
                and len(cached_lc) == len(lambdas_coul)
                and len(cached_lv) == len(lambdas_vdw)
                and np.allclose(cached_lc, lambdas_coul, atol=1e-9)
                and np.allclose(cached_lv, lambdas_vdw, atol=1e-9)
            )
            if not lambdas_match:
                print(
                    "  [WARN] IBS 状态对应的 λ 值与当前窗口不匹配（λ 路径已被自动加密/"
                    "重新划分窗口，或旧状态缺少 λ 元数据），完全忽略旧状态（不作为"
                    "热启动，f_k[k] 错配到不同 λ 上是主动引入偏差，不是中性起点），"
                    "从 f_k=0 重新开始"
                )
                return False

            current_score_sha256 = getattr(self, "sampling_score_sha256", None)
            cached_score_sha256 = state.get("sampling_score_sha256")
            if current_score_sha256 is not None and cached_score_sha256 != current_score_sha256:
                raise ExistingEnsembleRequiresRescueAudit(
                    "IBS state score identity mismatch: cached "
                    f"sampling_score_sha256={cached_score_sha256!r}, current="
                    f"{current_score_sha256!r}. Refusing to inject f_k or overwrite the "
                    "existing ensemble; route it to rescue/provenance audit.",
                    diagnostics={
                        "state_file": os.path.abspath(filepath),
                        "cached_sampling_score_sha256": cached_score_sha256,
                        "current_sampling_score_sha256": current_score_sha256,
                        "requires_rescue_audit": True,
                    },
                )
            if (
                current_score_sha256 is not None
                and state.get("residual_sampling_protocol_version")
                != IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION
            ):
                raise ExistingEnsembleRequiresRescueAudit(
                    "IBS state residual sampling protocol version is missing or "
                    "incompatible; refusing to inject f_k into the current ensemble.",
                    diagnostics={
                        "state_file": os.path.abspath(filepath),
                        "cached_residual_sampling_protocol_version": state.get(
                            "residual_sampling_protocol_version"
                        ),
                        "current_residual_sampling_protocol_version": (
                            IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION
                        ),
                        "requires_rescue_audit": True,
                    },
                )
            if current_score_sha256 is not None:
                cached_residual = state.get("residual_sampling")
                if not isinstance(cached_residual, dict) or cached_residual.get(
                    "em_policy"
                ) != "no_residual_twin":
                    raise ExistingEnsembleRequiresRescueAudit(
                        "IBS state residual EM/plugin identity is missing or incompatible; "
                        "refusing to inject f_k into the current ensemble.",
                        diagnostics={
                            "state_file": os.path.abspath(filepath),
                            "cached_residual_sampling": cached_residual,
                            "requires_rescue_audit": True,
                        },
                    )

            # The persisted tmbar_history is the actual u_kn history used to
            # continue bias learning.  A Stage 2 history sampled with the old
            # softcore Hamiltonian must never seed the current Hamiltonian;
            # Stage 1 charging states intentionally do not carry this gate.
            if (
                stage_type == "vdw"
                and state.get("vdw_nonbonded_protocol_version")
                != VDW_NONBONDED_PROTOCOL_VERSION
            ):
                print(
                    "  [WARN] Stage 2 IBS 状态的 "
                    f"vdw_nonbonded_protocol_version={state.get('vdw_nonbonded_protocol_version')!r} "
                    f"与当前 {VDW_NONBONDED_PROTOCOL_VERSION} 不匹配，"
                    "拒绝恢复旧 Hamiltonian 的 f_k/u_kn 历史"
                )
                return False

            # repair policy 是当前协议 ensemble 身份的一部分，但必须放在上面的
            # 协议版本/λ 门之后：旧版本 state 本来就应无条件失效，不能因为它在
            # 当时尚不存在的字段为 None 而被误判成“当前协议的可疑 ensemble”。
            # 对真正属于当前协议的 state，仍然在读取/注入任何 f_k 之前 fail-closed。
            current_repair_policy = getattr(
                self, "sampling_repair_policy", "non_mutating_v1"
            )
            cached_repair_policy = state.get("sampling_repair_policy")
            if cached_repair_policy != current_repair_policy:
                if current_repair_policy == "non_mutating_v1":
                    raise ExistingEnsembleRequiresRescueAudit(
                        "IBS 状态的 sampling_repair_policy="
                        f"{cached_repair_policy!r}（当前 {current_repair_policy!r}）。"
                        "non_mutating_v1 拒绝把当前协议下无策略/旧策略 state 的 f_k "
                        "注入 Context，也拒绝在原 ensemble 目录静默重新开始；"
                        "已保留原文件，交 rescue audit 判定。",
                        diagnostics={
                            "state_file": os.path.abspath(filepath),
                            "cached_sampling_repair_policy": cached_repair_policy,
                            "current_sampling_repair_policy": current_repair_policy,
                            "requires_rescue_audit": True,
                        },
                    )
                print(
                    "  [WARN] IBS 状态 sampling_repair_policy 不匹配 "
                    f"(cache={cached_repair_policy!r}, current={current_repair_policy!r})，"
                    "legacy_mutating 模式下忽略旧状态并重新开始"
                )
                return False

            f_k = state["f_k"]
            t = state["t"]
            if len(f_k) != self.n_states or not np.all(np.isfinite(np.asarray(f_k, dtype=float))):
                print("  [WARN] IBS 状态 f_k 无效，忽略旧状态")
                return False
            cached_eta_penalty = state.get("eta_penalty")
            if (
                cached_eta_penalty is None
                or not np.isfinite(float(cached_eta_penalty))
                or not (0.25 <= float(cached_eta_penalty) <= 1.0)
            ):
                print("  [WARN] IBS 状态 eta_penalty 无效，忽略旧状态")
                return False

            # [IBS_BIAS_PROTOCOL_VERSION=19] 每条 tmbar_history entry 必须完整
            # 校验（形状/有限性/lambda_indices 覆盖本窗口全部态/sampled_
            # distribution_row 恒为 0）才注入，任何一条不对就整份忽略——
            # 这份历史直接决定后续 solve_stage_integrated 的 f_k，绝不能
            # 部分信任、悄悄跳过坏 entry。
            cached_tmbar_history = state.get("tmbar_history")
            expected_lambda_indices = list(range(self.n_states))
            tmbar_history_valid = isinstance(cached_tmbar_history, list)
            validated_tmbar_history: List[Dict[str, Any]] = []
            if tmbar_history_valid:
                # Older checkpoints may predate the cap.  Only the newest
                # bounded suffix can influence future learning, so validate
                # and restore that suffix instead of re-inflating memory.
                cached_history_suffix = cached_tmbar_history[
                    -TMBAR_HISTORY_MAX_ENTRIES:
                ]
                for entry in cached_history_suffix:
                    try:
                        if not isinstance(entry, dict):
                            raise ValueError("entry 不是 dict")
                        if [int(x) for x in entry["lambda_indices"]] != expected_lambda_indices:
                            raise ValueError("lambda_indices 与当前窗口态数不匹配")
                        if int(entry.get("sampled_distribution_row", -1)) != 0:
                            raise ValueError("sampled_distribution_row 必须为 0")
                        u_kn = np.asarray(entry["u_kn"], dtype=np.float64)
                        bias_e = np.asarray(entry["bias_energies"], dtype=np.float64).ravel()
                        base_e = np.asarray(entry["base_energies"], dtype=np.float64).ravel()
                        if (
                            u_kn.ndim != 2
                            or u_kn.shape[0] != self.n_states
                            or u_kn.shape[1] < 1
                            or bias_e.size != u_kn.shape[1]
                            or base_e.size != u_kn.shape[1]
                            or not np.all(np.isfinite(u_kn))
                            or not np.all(np.isfinite(bias_e))
                            or not np.all(np.isfinite(base_e))
                        ):
                            raise ValueError("u_kn/bias_energies/base_energies 形状或有限性无效")
                        validated_tmbar_history.append({
                            "u_kn": u_kn,
                            "bias_energies": bias_e,
                            "base_energies": base_e,
                            "lambda_indices": expected_lambda_indices,
                            "sampled_distribution_row": 0,
                        })
                    except Exception:
                        tmbar_history_valid = False
                        break
            if not tmbar_history_valid:
                print("  [WARN] IBS 状态的 tmbar_history 无效，忽略旧状态")
                return False

            self.e_offset = state.get("e_offset", 0.0)
            self.bias_converged = bool(state.get("bias_converged", False))
            self.tmbar_history = validated_tmbar_history
            prior_dropped = state.get("tmbar_history_dropped_entries", 0)
            try:
                prior_dropped = max(0, int(prior_dropped))
            except (TypeError, ValueError):
                prior_dropped = 0
            self.tmbar_history_dropped_entries = (
                prior_dropped
                + max(0, len(cached_tmbar_history) - len(validated_tmbar_history))
            )
            self.eta_penalty = float(cached_eta_penalty)

            # 🔑 [2026-09-11] 账本：有就原样恢复；缺就从已保存的运行记录迁移并标记
            # 不完整（migrate_warmup_budget_ledger），**绝不默认已耗 0 步**。
            _cached_ledger = state.get("warmup_budget_ledger")
            if isinstance(_cached_ledger, dict):
                self.warmup_budget_ledger = dict(_cached_ledger)
            else:
                self.warmup_budget_ledger = migrate_warmup_budget_ledger(state, 0)
            self.f_k_evidence_status = str(
                state.get("f_k_evidence_status", F_K_EVIDENCE_NONE)
            )
            self.frozen_candidate_fingerprint = state.get("frozen_candidate_fingerprint")
            # 采样身份：给 skip_warmup_entirely 判"旧 PASS 还算不算数"用。
            self.loaded_stage_protocol_key = state.get("stage_protocol_key")
            self.loaded_stage_protocol_key_source = (
                "ibs_state" if self.loaded_stage_protocol_key is not None else None)
            if self.loaded_stage_protocol_key is None:
                # 🔑🔑 [2026-09-14] **既有 state 的身份恢复（可证明的那种）。**
                #
                # 写侧此前从来没把 `stage_protocol_key` 盖到 sampler 上，所以**所有
                # 已存在的 state 里这个字段都是 None** ⟹ 身份闸恒判"缺失" ⟹ 每次
                # resume 都重走一遍完整预热、白烧预热预算。只修写侧的话，现有 run
                # **还要再烧一轮**才能进入干净状态。
                #
                # 这里做的**不是**"缺了就当通过"，而是**从一份独立的、同一次生产写下
                # 的记录里把它取回来**：同窗口的 `convergence.json` 一直记着
                # `stage_protocol_key`（`_resume_cached_window_gate_status` 就是拿它
                # 跟当前身份比的）。取回来的前提是两份记录描述**同一个窗口几何** ——
                # λ 必须逐位相等，否则那份 convergence 属于另一套布局，不能借。
                #
                # ⚠️ 仍然 fail-closed：取不到、或 λ 对不上 ⟹ 保持 None，
                # 照旧重走预热。**绝不凭空合成一个身份。**
                _recovered, _src = _recover_stage_identity_from_convergence(
                    filepath, state, lambdas_coul, lambdas_vdw, stage_type)
                if _recovered is not None:
                    self.loaded_stage_protocol_key = _recovered
                    self.loaded_stage_protocol_key_source = _src
                    print(
                        f"  [INFO] IBS 状态里没有采样身份（写侧历史遗漏），已从 {_src} "
                        "取回 —— 两份记录的 λ 逐位相等，属同一窗口几何。"
                        "旧 PASS 因此仍需按**当前**身份比对，不是无条件放行。"
                    )

            cached_status = state.get("bias_status", "unconverged")
            cached_pending_f_k = state.get("frozen_f_k_pending")
            if (
                cached_status == "calibrated_pending_validation"
                and cached_pending_f_k is not None
                and len(cached_pending_f_k) == self.n_states
                and np.all(np.isfinite(np.asarray(cached_pending_f_k, dtype=float)))
            ):
                self.bias_status = "calibrated_pending_validation"
                self.frozen_f_k_pending = [float(x) for x in cached_pending_f_k]
                self.frozen_validation_cumulative_steps = int(
                    state.get("frozen_validation_cumulative_steps", 0)
                )
            elif (
                cached_status == "frozen_validation_indeterminate"
                and cached_pending_f_k is not None
                and len(cached_pending_f_k) == self.n_states
                and np.all(np.isfinite(np.asarray(cached_pending_f_k, dtype=float)))
            ):
                # 🔑 [2026-09-11] 上次是"验证预算耗尽、无法判定"：对 f_k 没有任何
                # 反面证据，所以既不重新 SGD、也不当成终态。恢复同一份冻结 f_k、
                # 同一份已耗批数/步数，接着验。
                self.bias_status = "frozen_validation_indeterminate"
                self.frozen_f_k_pending = [float(x) for x in cached_pending_f_k]
                self.frozen_validation_cumulative_steps = int(
                    state.get("frozen_validation_cumulative_steps", 0)
                )
                self.frozen_validation_batches_done = int(
                    state.get("frozen_validation_batches_done", 0)
                )
            elif cached_status == "calibrated_validation_failed":
                # 🔑 终态：冻结验证累计预算已经用到最后一档仍未通过，不再是
                # "pending"——不能把这份 f_k 当成可以继续自动续验的东西注入，
                # run_all_windows 里紧接着 load_ibs_state 之后的检查会看到这个
                # 状态并立刻硬停止，不做任何进一步的 SGD/续验尝试。
                self.bias_status = "calibrated_validation_failed"
                self.frozen_f_k_pending = None
                self.frozen_validation_cumulative_steps = 0
            elif cached_status == "provisional_production":
                # 🔑 [2026-09-15] 临时生产：这份 f_k 的冻结验证撞了 15 批上限、
                # Δf−ΔF **从未被求出**（无结论，既非通过也非被驳回），调用方
                # 显式授权用它跑**一块**诊断生产。该窗口已经有生产产物，
                # 恢复时既不重新 learning、也不重新验证（那会把已经花掉的
                # Epoch 再烧一遍），但**绝不**升级成 verified。
                # `frozen_f_k_pending` 必须是 None：否则 run_all_windows 那道
                # non_mutating_v1 的 ExistingEnsembleRequiresRescueAudit 会把
                # 它当成旧变异路径的产物拒掉。
                self.bias_status = "provisional_production"
                self.frozen_f_k_pending = None
                self.frozen_validation_cumulative_steps = int(
                    state.get("frozen_validation_cumulative_steps", 0)
                )
            elif cached_status == "failed":
                # [Candidate-first, Validate-or-Learn v1] 新协议下的终态失败
                # 值——必须原样保留，不能落进下面的 else 分支被静默改写成
                # "unconverged"，否则 run_all_windows 里那一处专门识别
                # "failed" 为终态的检查（10586 行附近）永远看不到这个值，
                # 一份已经判定终态失败的窗口会被当成普通未收敛热启动重新学习。
                self.bias_status = "failed"
                self.frozen_f_k_pending = None
                self.frozen_validation_cumulative_steps = 0
            else:
                self.bias_status = "converged" if self.bias_converged else "unconverged"
                self.frozen_f_k_pending = None
                self.frozen_validation_cumulative_steps = 0

            # [Candidate-first, Validate-or-Learn v1] 纯附加元数据恢复，缺失
            # 时安全回退默认值，绝不新增 fail-closed 规则。一次成功的
            # load_ibs_state 按定义就是一次 resume——seed_source 在这里写
            # "resume"，而不是在本次刻意不重写的 10586-10689 resume 路由分支
            # 里，两者等价，但避免了对那段控制流的任何改动。
            self.seed_source = "resume"
            self.validation_attempts = int(state.get("validation_attempts", 0) or 0)
            self.last_failure_reason = state.get("last_failure_reason")

            # 🔑 [2026-08-27] 恢复生产入口标记。缺失（旧文件）或形状/有限性
            # 校验不过时保持 None（"历史入口未知"），绝不能拿这次 resume
            # 之后才会算出的值冒充最初进入生产时的值——那正是这个字段存在
            # 的意义。
            cached_production_entry_f_k = state.get("production_entry_f_k")
            if (
                cached_production_entry_f_k is not None
                and len(cached_production_entry_f_k) == self.n_states
                and np.all(np.isfinite(np.asarray(cached_production_entry_f_k, dtype=float)))
            ):
                self.production_entry_f_k = [float(x) for x in cached_production_entry_f_k]
            else:
                self.production_entry_f_k = None

            # 注入 Context
            for k in range(self.n_states):
                self.context.setParameter(f"{self.prefix}_f_{k}", float(f_k[k]))

            cached_f_history = state.get("f_history_kj_mol")
            valid_f_history = []
            if isinstance(cached_f_history, list) and len(cached_f_history) == int(t):
                for values in cached_f_history:
                    array = np.asarray(values, dtype=np.float64)
                    if array.shape != (self.n_states,) or not np.all(np.isfinite(array)):
                        valid_f_history = []
                        break
                    valid_f_history.append(array.copy())
            # Backward compatible: old checkpoints preserve the learning-rate
            # update count; new checkpoints retain the actual trajectory for
            # restart continuity and EXP-030 audit.
            self.f_history = (
                valid_f_history if len(valid_f_history) == int(t)
                else [np.array(f_k, dtype=np.float64)] * int(t)
            )
            print(
                f"  IBS 状态已恢复: t={t}, max|f_k|={np.max(np.abs(f_k)):.2f} kJ/mol, "
                f"tmbar_history entries={len(self.tmbar_history)}, "
                f"bias_converged={self.bias_converged}, bias_status={self.bias_status}"
            )
            return True
        except ExistingEnsembleRequiresRescueAudit:
            raise
        except Exception as e:
            print(f"  [WARN] IBS 状态加载失败: {e}")
            return False
def _compute_bidirectional_overlap_from_u_kn(
    u_kn: np.ndarray,
    n_k: np.ndarray,
    threshold: float = 0.03,
) -> Dict[str, Any]:
    """Run the repository's normal MBAR overlap calculation for two states."""
    if not has_pymbar():
        raise ImportError("fixed-lambda overlap 探针需要 pymbar")
    u_kn = np.asarray(u_kn, dtype=np.float64)
    n_k = np.asarray(n_k, dtype=int)
    if u_kn.ndim != 2 or u_kn.shape[0] != 2 or n_k.shape != (2,):
        raise ValueError(f"双态 MBAR 输入形状错误: u_kn={u_kn.shape}, n_k={n_k.shape}")
    if int(np.sum(n_k)) != u_kn.shape[1] or np.any(n_k < 2):
        raise ValueError(f"双态 MBAR 样本计数错误: u_kn={u_kn.shape}, n_k={n_k.tolist()}")
    valid = np.all(np.isfinite(u_kn), axis=0)
    if not np.all(valid):
        raise RuntimeError("fixed-lambda overlap 探针产生 NaN/Inf，拒绝把坏帧静默丢弃")
    stable = u_kn - np.min(u_kn, axis=0, keepdims=True)
    mbar = _build_mbar_compatible(
        stable,
        n_k,
        relative_tolerance=1e-7,
        initialize="BAR",
        solver_protocol="default",
    )
    overlap_matrix = np.asarray(mbar.compute_overlap()["matrix"], dtype=np.float64)
    if overlap_matrix.shape != (2, 2) or not np.all(np.isfinite(overlap_matrix)):
        raise RuntimeError(f"PyMBAR 返回无效 overlap matrix: {overlap_matrix}")
    minimum = float(min(overlap_matrix[0, 1], overlap_matrix[1, 0]))
    # 🔑 [IBS_BIAS_PROTOCOL_VERSION=8] MBAR 在算 overlap 的同时已经把这两态之间的
    # BAR/MBAR 自由能差解出来了——这正是 IBS 偏置 U_k-f_k 要求的 f_k（均匀混合
    # 需要 f_k 等于状态 k 的相对自由能）。overlap 全部通过但 SGD 学习一直振荡时，
    # 与其继续在一个不稳定的求解器里盲搜，不如直接用这个已经算出来的真值校准
    # f_k，见调用方 run_all_windows 里 fixed-H overlap 全通过后的处理。返回的是
    # 约化单位（kT），换算成 kJ/mol 由调用方用自己的 kt 完成。
    res = _compute_free_energy_result_compatible(mbar, compute_uncertainty=True)
    df_matrix, ddf_matrix = _extract_free_energy_arrays(res, require_uncertainty=True)
    delta_f_reduced = float(df_matrix[0, 1])
    delta_f_uncertainty_reduced = float(ddf_matrix[0, 1])
    if not np.isfinite(delta_f_reduced):
        raise RuntimeError(f"PyMBAR 返回非有限的双态自由能差: {delta_f_reduced}")
    # 🔑 之前只检查了 delta_f 本身，没检查它的不确定度。这个不确定度会被下游
    # abfe_pipeline.py::_diagnose_and_repair_all_pass_low_ess_window 直接当噪声
    # 尺度用在判据 max(floor, sigma_multiplier * sigma) 里——如果 sigma 是 NaN，
    # 这个阈值也会变成 NaN，Python 的 `>` 比较对 NaN 恒为 False，会让"任意差异都
    # 判定为在阈值内"，错误地把本该判定为 recalibrate_f_k 的窗口放行成
    # reseed_resample。同样，负的不确定度在物理上没有意义，说明 PyMBAR 求解或
    # 上游数据有问题，不应该被静默当成正常值传下去。两种情况都必须 fail closed。
    if not (np.isfinite(delta_f_uncertainty_reduced) and delta_f_uncertainty_reduced >= 0.0):
        raise RuntimeError(
            f"PyMBAR 返回无效的双态自由能差不确定度: {delta_f_uncertainty_reduced}"
            "（非有限或为负），拒绝把它当作噪声尺度使用。"
        )
    return {
        "overlap_matrix": overlap_matrix.tolist(),
        "min_bidirectional_overlap": minimum,
        "threshold": float(threshold),
        "passed": bool(minimum >= float(threshold)),
        "delta_f_reduced_kT": delta_f_reduced,
        "delta_f_uncertainty_reduced_kT": delta_f_uncertainty_reduced,
    }


def probe_bidirectional_overlap(
    topology,
    common_system_xml: str,
    ibs_wrapper,
    state_i: int,
    state_j: int,
    positions,
    box_vectors,
    temperature,
    platform_name: str,
    burn_in_steps: int = 5000,
    sample_steps: int = 20000,
    sample_interval: int = 500,
    threshold: float = 0.03,
) -> Dict[str, Any]:
    """Measure overlap from independent unbiased fixed-H trajectories at i and j.

    Each dynamics system is exactly ``U_common + cv_k_int``.  It contains no
    CustomCVForce, no IBS bias and no WCA sampling shell.  The two trajectories
    are NVT because the common window system has no barostat and uses the same
    LangevinMiddleIntegrator settings as production.  Energies of both target
    states are evaluated at every frame through IBSSampler's existing probe
    context, then passed to the normal PyMBAR overlap implementation.
    """
    if state_i == state_j:
        raise ValueError("fixed-lambda overlap 的两个状态必须不同")
    cv_xmls = list(getattr(ibs_wrapper, "_int_cv_force_xmls", []))
    if max(state_i, state_j) >= len(cv_xmls):
        raise RuntimeError("IBS wrapper 缺少 fixed-lambda overlap 所需的 CV XML")
    if sample_interval <= 0 or sample_steps < 2 * sample_interval:
        raise ValueError("fixed-lambda overlap 采样长度不足")
    temperature_q = (
        temperature
        if hasattr(temperature, "value_in_unit")
        else float(temperature) * unit.kelvin
    )

    resolved_platform, props = _build_platform_properties(platform_name)
    platform = openmm.Platform.getPlatformByName(resolved_platform)
    simulations = []

    def _build_fixed_simulation(state_index: int, seed_offset: int):
        fixed_system = ensure_owned_system(XmlSerializer.deserialize(common_system_xml))
        fixed_cv = XmlSerializer.deserialize(cv_xmls[state_index])
        fixed_cv.setForceGroup(1)
        fixed_system.addForce(fixed_cv)
        if any(int(fixed_system.getForce(k).getForceGroup()) == 4 for k in range(fixed_system.getNumForces())):
            raise RuntimeError("fixed-lambda dynamics system 中不应存在 WCA group 4")
        integrator = LangevinMiddleIntegrator(
            temperature_q, 2.0 / unit.picosecond, 0.002 * unit.picosecond
        )
        integrator.setConstraintTolerance(1e-3)
        integrator.setRandomNumberSeed(8731 + int(seed_offset))
        if hasattr(integrator, "setRemoveCMMotion"):
            integrator.setRemoveCMMotion(True)
        simulation = app.Simulation(topology, fixed_system, integrator, platform, props)
        if box_vectors is not None:
            simulation.context.setPeriodicBoxVectors(*box_vectors)
        simulation.context.setPositions(positions)
        # 🔑 [Hamiltonian mismatch fix] Context 默认 lambda_boresch_scale=0（见
        # LambdaDependentBoreschForce.__init__），但主窗口生产轨迹早已爬坡到
        # 1.0——必须显式对齐，否则这个 fixed-H 探针评估的是关掉 Boresch 限制的
        # 系统，跟它要验证的生产 Hamiltonian 不是同一个。
        if _system_has_global_parameter(fixed_system, "lambda_boresch_scale"):
            simulation.context.setParameter("lambda_boresch_scale", 1.0)
        simulation.context.setVelocitiesToTemperature(temperature_q, 29173 + int(seed_offset))
        return simulation

    def _collect_frames(simulation):
        guarded_step(simulation, int(burn_in_steps), "重叠探针 burn-in")
        frames = []
        n_samples = int(sample_steps) // int(sample_interval)
        for _ in range(n_samples):
            guarded_step(simulation, int(sample_interval), "重叠探针采样")
            state = simulation.context.getState(getPositions=True)
            # 🔑 [2026-09-09] `asNumpy=True`：不加时 getPositions() 返回的是一个
            # 装着 N 个 Python `Vec3` 对象的 list —— 73k 原子 ≈ 每帧几百万个小对象、
            # 约 10 MB，而这里要把**两个系综的全部帧**都囤起来再统一求能量
            # （校准探针的 sample_steps 会按 20000→40000→80000 阶梯重试，
            # 单条边峰值可达数百帧）。obmalloc 的 arena 之后也不还给 OS。
            # 新的 bank 路径（`_extend_state_trajectory`）是逐帧求值只留 K 个 float，
            # 这两个遗留探针没跟上。改成 ndarray 后同样的帧数占用降一个量级，
            # 且 `setPositions` 接受 Quantity(ndarray)，下游行为不变。
            frames.append(
                (state.getPositions(asNumpy=True), state.getPeriodicBoxVectors())
            )
        return frames

    # 🔑 [2026-09-09] 预置成 None，好让下面的 finally 能无条件断引用
    # （原来 finally 只 clear 那个 list，见其中的说明）。
    sim_i = sim_j = evaluator = frames_by_ensemble = None
    try:
        sim_i = _build_fixed_simulation(state_i, 0)
        simulations.append(sim_i)
        sim_j = _build_fixed_simulation(state_j, 1)
        simulations.append(sim_j)
        frames_by_ensemble = [_collect_frames(sim_i), _collect_frames(sim_j)]

        # Reuse the already validated all-state energy-only probe machinery.
        evaluator = IBSSampler(
            sim_i.context,
            len(cv_xmls),
            temperature_q,
            prefix=getattr(ibs_wrapper, "prefix", "abfe"),
            ibs_wrapper=ibs_wrapper,
        )
        # 🔑 跟 IBSSampler._lj_tail_correction_kj_mol() 读的是同一个
        # ibs_wrapper.lj_tail_lrc_coeff_kj_mol（switching+softcore-aware，见
        # _lj_tail_lrc_coefficients_kj_mol），保证 fixed-H overlap 探针和
        # 生产采样对同一个 λ 用同一组 LRC 系数，不会出现两条路径各算一遍、
        # 结果不一致的情况。
        lrc_coeff = getattr(ibs_wrapper, "lj_tail_lrc_coeff_kj_mol", None)
        selected = np.asarray([state_i, state_j], dtype=int)
        reduced_by_ensemble = []
        inefficiencies = []
        kt = (unit.MOLAR_GAS_CONSTANT_R * temperature_q).value_in_unit(unit.kilojoule_per_mole)
        beta = 1.0 / float(kt)
        for frames in frames_by_ensemble:
            values = []
            for frame_positions, frame_box in frames:
                e_kj = evaluator.evaluate_interaction_energies(frame_positions, frame_box)[selected]
                if lrc_coeff is not None:
                    box_nm = _box_vectors_to_nm_array(frame_box)
                    volume_nm3 = abs(float(np.linalg.det(box_nm)))
                    if not np.isfinite(volume_nm3) or volume_nm3 <= 0.0:
                        raise RuntimeError("fixed-lambda overlap 帧的盒子体积无效")
                    e_kj = e_kj + np.asarray(lrc_coeff, dtype=np.float64)[selected] / volume_nm3
                values.append(beta * np.asarray(e_kj, dtype=np.float64))
            reduced = np.asarray(values, dtype=np.float64).T
            decorrelated, g = subsample_series_by_autocorrelation(reduced[1] - reduced[0])
            reduced_by_ensemble.append(reduced[:, decorrelated])
            inefficiencies.append(float(g))

        n_k = np.asarray([block.shape[1] for block in reduced_by_ensemble], dtype=int)
        u_kn = np.concatenate(reduced_by_ensemble, axis=1)
        result = _compute_bidirectional_overlap_from_u_kn(u_kn, n_k, threshold=threshold)
        result.update({
            "local_states": [int(state_i), int(state_j)],
            "n_k_decorrelated": n_k.tolist(),
            "statistical_inefficiency": inefficiencies,
            "burn_in_steps": int(burn_in_steps),
            "sample_steps": int(sample_steps),
            "sample_interval": int(sample_interval),
            "dynamics_hamiltonian": "U_common_plus_single_cv_int",
            "ensemble": "NVT",
            # kJ/mol version of the BAR/MBAR free energy difference already
            # solved above. This is the *physical* delta_f (WCA-less
            # dynamics, LRC included in the energies fed to MBAR) -- useful
            # for path/lambda-grid overlap diagnostics, but
            # [IBS_BIAS_PROTOCOL_VERSION=10] it must NOT be used to calibrate
            # the IBS bias CV's f_k: production actually samples
            # U_common + WCA_window(lambda_shield) + CV_k (this function's
            # dynamics omit the WCA shield entirely), and the CV's own energy
            # never includes LRC (that is only added afterward, offline, to
            # target_energies feeding MBAR -- see IBSSampler.collect_energies).
            # Calibrating f_k from this field was a real, confirmed bug: the
            # measured delta_f corresponds to a different ensemble and a
            # different energy definition than what the bias force actually
            # needs to reproduce a uniform mixture over. Use
            # probe_bidirectional_overlap_for_bias_calibration's
            # delta_f_bias_kJ_mol for that purpose instead.
            "delta_f_kJ_mol": result["delta_f_reduced_kT"] * kt,
            "delta_f_uncertainty_kJ_mol": result["delta_f_uncertainty_reduced_kT"] * kt,
        })
        return result
    finally:
        # 🔑 [2026-09-09] 原来只有 `simulations.clear()` —— 那是**空转**：
        # sim_i / sim_j / frames_by_ensemble 仍然绑在本函数栈帧上，而
        # evaluator.context 还是 sim_i.context 的第三个引用。于是两个动力学
        # Context + 一个探针 Context + 整堆采样帧一直活到栈帧销毁，而不是被
        # 这句 gc.collect() 回收。本文件其它所有持有 Context 的地方都是
        # `del sim; gc.collect()` 的写法，bank 变体还额外做 `evaluator.context
        # = None`（那里的注释写的就是这个隐患），只有这两个遗留探针两样都没做。
        if evaluator is not None and hasattr(evaluator, "context"):
            evaluator.context = None
        sim_i = sim_j = evaluator = frames_by_ensemble = None
        simulations.clear()
        gc.collect()


def probe_bidirectional_overlap_for_bias_calibration(
    topology,
    common_plus_wca_system_xml: str,
    ibs_wrapper,
    state_i: int,
    state_j: int,
    positions,
    box_vectors,
    temperature,
    platform_name: str,
    lambda_shield: float,
    burn_in_steps: int = 5000,
    sample_steps: int = 20000,
    sample_interval: int = 500,
    min_decorrelated_samples: int = 20,
    max_delta_f_uncertainty_kJ_mol: float = 1.0,
    overlap_threshold: float = 0.03,
) -> Optional[Dict[str, Any]]:
    """Measure the adjacent-state free energy difference under the *exact*
    ensemble and energy definition the IBS bias CV's f_k actually needs to
    reproduce a uniform mixture over -- deliberately a separate function from
    ``probe_bidirectional_overlap`` (used elsewhere for path/lambda-grid
    overlap diagnostics), not a mode flag on it, so the two purposes can
    never again be silently conflated the way they were before
    [IBS_BIAS_PROTOCOL_VERSION=10]:

    - Dynamics: ``U_common + WCA_window(lambda_shield) + CV_k`` (Group 1
      single-state), built from ``common_plus_wca_system_xml`` (see
      ``_serialize_ibs_common_plus_wca_system``) with ``lambda_shield`` set
      to the same value production used for this window
      (``mean(lambdas_vdw_in_window)``). The WCA shield only cancels out of
      same-frame, same-window energy *differences* between states; it does
      not disappear from the sampled trajectory, so omitting it (as the
      path-overlap probe deliberately does, for its own different purpose)
      samples a genuinely different conformational ensemble than production.
    - Energy: raw softcore CV interaction energy only (``IBSSampler.
      evaluate_interaction_energies``), with **no** long-range dispersion
      correction added. Production's Group 1 bias force never includes LRC
      either -- LRC is only added afterward, offline, to ``target_energies``
      feeding MBAR/final analysis (see ``IBSSampler.collect_energies``:
      ``bias_cv_energies`` vs. ``target_energies = softcore_energies +
      lrc_energies``). Calibrating the bias CV from an LRC-inclusive delta_f
      would inject an energy term the bias force itself never applies.

    Returns ``None`` -- not a partial/best-effort result -- if, after
    ``burn_in_steps + sample_steps``, either the decorrelated sample count
    (``n_k_decorrelated``, per the *smaller* of the two ensembles) is below
    ``min_decorrelated_samples`` or the resulting delta_f uncertainty exceeds
    ``max_delta_f_uncertainty_kJ_mol``. A real overlap threshold gate alone
    (``passed``/``min_bidirectional_overlap`` are still reported for
    diagnostics) is not sufficient evidence that delta_f is precise enough to
    directly overwrite f_k -- e.g. a real run here saw
    ``n_k_decorrelated=[39, 9]`` with ``overlap=0.105``: minimally connected,
    but far too few decorrelated samples on one side to trust the resulting
    delta_f as a calibration target. The caller must extend sampling (larger
    ``sample_steps``) and retry rather than accept and use an
    under-supported estimate.
    """
    if state_i == state_j:
        raise ValueError("bias 校准探针的两个状态必须不同")
    cv_xmls = list(getattr(ibs_wrapper, "_int_cv_force_xmls", []))
    if max(state_i, state_j) >= len(cv_xmls):
        raise RuntimeError("IBS wrapper 缺少 bias 校准探针所需的 CV XML")
    if sample_interval <= 0 or sample_steps < 2 * sample_interval:
        raise ValueError("bias 校准探针采样长度不足")
    temperature_q = (
        temperature
        if hasattr(temperature, "value_in_unit")
        else float(temperature) * unit.kelvin
    )

    resolved_platform, props = _build_platform_properties(platform_name)
    platform = openmm.Platform.getPlatformByName(resolved_platform)
    simulations = []
    # 🔑 0.5→1.0→2.0 fs 短 ramp，2.0 fs 正好是正式 burn-in/采样步长，ramp 结束后
    # 不需要再显式恢复步长。步数很小（三段共 400 步 ≈ 0.5 ps），相对
    # burn_in_steps（默认 5000 步 = 10 ps）+ sample_steps 的开销可以忽略。
    _RAMP_STAGES_FS_STEPS = ((0.5, 100), (1.0, 100), (2.0, 200))

    def _finite_check(simulation, label: str) -> None:
        state = simulation.context.getState(getEnergy=True, getForces=True, getPositions=True)
        e_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        positions_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        if not (np.isfinite(e_kj) and np.all(np.isfinite(forces)) and np.all(np.isfinite(positions_nm))):
            raise RuntimeError(
                f"[bias 校准探针] {label}：检测到非有限的能量/力/坐标"
                f"（potential_energy={e_kj!r} kJ/mol）。提前终止，不进入正式 "
                "burn-in/采样。"
            )

    def _build_fixed_simulation(state_index: int, seed_offset: int):
        label = f"state_index={state_index}（edge state_i={state_i}, state_j={state_j}）"
        fixed_system = ensure_owned_system(XmlSerializer.deserialize(common_plus_wca_system_xml))
        fixed_cv = XmlSerializer.deserialize(cv_xmls[state_index])
        fixed_cv.setForceGroup(1)
        fixed_system.addForce(fixed_cv)
        has_group4 = any(
            int(fixed_system.getForce(k).getForceGroup()) == 4
            for k in range(fixed_system.getNumForces())
        )
        # 🔑 [WCA_SHIELD_RETIRED，2026-09-02] 同上，方向反转：探针必须与生产同系综。
        if WCA_SHIELD_RETIRED:
            if has_group4:
                raise RuntimeError(
                    "WCA_SHIELD_RETIRED=True 但 bias 校准探针的 fixed-H 系统里仍有 "
                    "Group 4 —— 生产不含防护壳，探针含壳会采到生产从不采的系综。"
                )
        elif not has_group4:
            raise RuntimeError("bias 校准探针的 fixed-H 系统缺少 Group 4 WCA 防护壳")
        integrator = LangevinMiddleIntegrator(
            temperature_q, 2.0 / unit.picosecond, 0.002 * unit.picosecond
        )
        integrator.setConstraintTolerance(1e-3)
        integrator.setRandomNumberSeed(41337 + int(seed_offset))
        if hasattr(integrator, "setRemoveCMMotion"):
            integrator.setRemoveCMMotion(True)
        simulation = app.Simulation(topology, fixed_system, integrator, platform, props)
        if box_vectors is not None:
            simulation.context.setPeriodicBoxVectors(*box_vectors)
        simulation.context.setPositions(positions)
        if _system_has_global_parameter(fixed_system, "lambda_shield"):
            simulation.context.setParameter("lambda_shield", float(lambda_shield))
        # 🔑 [Hamiltonian mismatch fix] 同 probe_bidirectional_overlap 里
        # _build_fixed_simulation 的注释：必须显式对齐到主窗口早已爬坡到的 1.0，
        # 否则这份"证明 f_k 物理正确"的校准探针实际验证的是另一个 Hamiltonian。
        if _system_has_global_parameter(fixed_system, "lambda_boresch_scale"):
            simulation.context.setParameter("lambda_boresch_scale", 1.0)
        # 🔑 [live crash fix] 在真实 lambda_shield + 这个态自己的 CV 力都已加入
        # 之后，针对这个具体态的精确 Hamiltonian 再单独最小化一次——外层调用方
        # 传入的 positions 只是这个窗口的粗弛豫构型（可能是在另一个
        # lambda_shield 值、甚至完全没有 WCA 防护壳的势能面上做的），不能保证
        # 对这个态、这个 lambda_shield 强度也是无碰撞的。这是防止防护壳力在
        # 动力学开始时撞上残余原子重叠、第一步就爆炸成 NaN 的最可靠手段
        # （比只保证外层粗弛豫时 lambda_shield 同步更进一步的第二道保险）。
        try:
            simulation.minimizeEnergy(maxIterations=2000)
        except openmm.OpenMMException as exc:
            raise RuntimeError(
                f"[bias 校准探针] {label}：局部最小化过程中出现非有限坐标"
                f"（原始异常: {exc}）。"
            ) from exc
        _finite_check(simulation, f"{label} 局部最小化后")
        simulation.context.setVelocitiesToTemperature(temperature_q, 62143 + int(seed_offset))
        # 🔑 最小化只保证落在一个（局部）能量极小点，不保证 Langevin 动力学
        # 用生产步长（2 fs）起步时力已经足够温和——用更小的步长先走几百步，
        # 每段之间检查有限性，再逐步过渡到正式步长，避免真实运行中窗口 (5,9)
        # 边 [7,8] 那种"第一步就 NaN"的爆炸；NaN 时明确报告是哪个 state/edge
        # 出的问题，而不是让 OpenMM 的裸 traceback 直接冒出来。
        for ramp_dt_fs, ramp_steps in _RAMP_STAGES_FS_STEPS:
            simulation.integrator.setStepSize(ramp_dt_fs * unit.femtosecond)
            guarded_step(simulation, ramp_steps, f"[bias 校准探针] {label} ramp {ramp_dt_fs} fs")
            _finite_check(simulation, f"{label} ramp {ramp_dt_fs} fs 后")
        return simulation

    def _collect_frames(simulation, state_index: int):
        label = f"state_index={state_index}（edge state_i={state_i}, state_j={state_j}）"
        guarded_step(simulation, burn_in_steps, f"[bias 校准探针] {label} burn-in")
        _finite_check(simulation, f"{label} burn-in 后")
        frames = []
        n_samples = int(sample_steps) // int(sample_interval)
        for _ in range(n_samples):
            guarded_step(simulation, sample_interval, f"[bias 校准探针] {label} 采样")
            state = simulation.context.getState(getPositions=True)
            # 🔑 [2026-09-09] `asNumpy=True`：不加时 getPositions() 返回的是一个
            # 装着 N 个 Python `Vec3` 对象的 list —— 73k 原子 ≈ 每帧几百万个小对象、
            # 约 10 MB，而这里要把**两个系综的全部帧**都囤起来再统一求能量
            # （校准探针的 sample_steps 会按 20000→40000→80000 阶梯重试，
            # 单条边峰值可达数百帧）。obmalloc 的 arena 之后也不还给 OS。
            # 新的 bank 路径（`_extend_state_trajectory`）是逐帧求值只留 K 个 float，
            # 这两个遗留探针没跟上。改成 ndarray 后同样的帧数占用降一个量级，
            # 且 `setPositions` 接受 Quantity(ndarray)，下游行为不变。
            frames.append(
                (state.getPositions(asNumpy=True), state.getPeriodicBoxVectors())
            )
        return frames

    # 🔑 [2026-09-09] 预置成 None，好让下面的 finally 能无条件断引用
    # （原来 finally 只 clear 那个 list，见其中的说明）。
    sim_i = sim_j = evaluator = frames_by_ensemble = None
    try:
        sim_i = _build_fixed_simulation(state_i, 0)
        simulations.append(sim_i)
        sim_j = _build_fixed_simulation(state_j, 1)
        simulations.append(sim_j)
        frames_by_ensemble = [
            _collect_frames(sim_i, state_i),
            _collect_frames(sim_j, state_j),
        ]

        evaluator = IBSSampler(
            sim_i.context,
            len(cv_xmls),
            temperature_q,
            prefix=getattr(ibs_wrapper, "prefix", "abfe"),
            ibs_wrapper=ibs_wrapper,
        )
        selected = np.asarray([state_i, state_j], dtype=int)
        reduced_by_ensemble = []
        inefficiencies = []
        kt = (unit.MOLAR_GAS_CONSTANT_R * temperature_q).value_in_unit(unit.kilojoule_per_mole)
        beta = 1.0 / float(kt)
        for frames in frames_by_ensemble:
            values = []
            for frame_positions, frame_box in frames:
                # 🔑 只用纯 softcore CV 能量，绝不加 LRC——见函数 docstring。
                e_kj = evaluator.evaluate_interaction_energies(frame_positions, frame_box)[selected]
                values.append(beta * np.asarray(e_kj, dtype=np.float64))
            reduced = np.asarray(values, dtype=np.float64).T
            decorrelated, g = subsample_series_by_autocorrelation(reduced[1] - reduced[0])
            reduced_by_ensemble.append(reduced[:, decorrelated])
            inefficiencies.append(float(g))

        n_k = np.asarray([block.shape[1] for block in reduced_by_ensemble], dtype=int)
        if int(np.min(n_k)) < int(min_decorrelated_samples):
            return None
        u_kn = np.concatenate(reduced_by_ensemble, axis=1)
        result = _compute_bidirectional_overlap_from_u_kn(u_kn, n_k, threshold=overlap_threshold)
        delta_f_bias_kj = result["delta_f_reduced_kT"] * kt
        delta_f_bias_uncertainty_kj = result["delta_f_uncertainty_reduced_kT"] * kt
        if delta_f_bias_uncertainty_kj > float(max_delta_f_uncertainty_kJ_mol):
            return None
        result.update({
            "local_states": [int(state_i), int(state_j)],
            "n_k_decorrelated": n_k.tolist(),
            "statistical_inefficiency": inefficiencies,
            "burn_in_steps": int(burn_in_steps),
            "sample_steps": int(sample_steps),
            "sample_interval": int(sample_interval),
            "dynamics_hamiltonian": "U_common_plus_wca_window_plus_single_cv_int_no_lrc",
            "ensemble": "NVT",
            "lambda_shield": float(lambda_shield),
            # These are the ONLY fields that may be used to calibrate f_k --
            # see docstring. Deliberately named differently from
            # probe_bidirectional_overlap's delta_f_kJ_mol so the two can
            # never be silently swapped again.
            "delta_f_bias_kJ_mol": delta_f_bias_kj,
            "delta_f_bias_uncertainty_kJ_mol": delta_f_bias_uncertainty_kj,
        })
        return result
    finally:
        # 🔑 [2026-09-09] 原来只有 `simulations.clear()` —— 那是**空转**：
        # sim_i / sim_j / frames_by_ensemble 仍然绑在本函数栈帧上，而
        # evaluator.context 还是 sim_i.context 的第三个引用。于是两个动力学
        # Context + 一个探针 Context + 整堆采样帧一直活到栈帧销毁，而不是被
        # 这句 gc.collect() 回收。本文件其它所有持有 Context 的地方都是
        # `del sim; gc.collect()` 的写法，bank 变体还额外做 `evaluator.context
        # = None`（那里的注释写的就是这个隐患），只有这两个遗留探针两样都没做。
        if evaluator is not None and hasattr(evaluator, "context"):
            evaluator.context = None
        sim_i = sim_j = evaluator = frames_by_ensemble = None
        simulations.clear()
        gc.collect()



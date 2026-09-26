@dataclass(frozen=True)
class OuterLambdaController:
    """协议 v1 的外层 λ 控制器。

    使用 :meth:`from_mapping` 构造可获得完整 fail-closed 配置验证。直接构造仍会在
    ``__post_init__`` 中验证数学不变量。
    """

    enabled: bool
    stage: str
    baseline_potential: str
    endpoint_tolerance: float
    coefficients: tuple[float, ...]
    max_abs_coefficient: float
    bases: tuple[NeuralBasisModelSpec, ...] = ()
    safety: NeuralPathSafety | None = None
    protocol_version: int = NEURAL_PATH_PROTOCOL_VERSION
    envelope_type: str = "sin2"
    coefficient_model_type: str = "constant"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise NeuralPathConfigError("enabled 必须是 bool")
        if self.protocol_version != NEURAL_PATH_PROTOCOL_VERSION:
            raise NeuralPathConfigError(
                f"不支持 neural path protocol_version={self.protocol_version}；"
                f"当前只支持 {NEURAL_PATH_PROTOCOL_VERSION}"
            )
        if self.envelope_type != "sin2":
            raise NeuralPathConfigError("协议 v1 只支持 envelope.type='sin2'")
        if self.coefficient_model_type != "constant":
            raise NeuralPathConfigError(
                "协议 v1 只支持 coefficient_model.type='constant'"
            )
        if not self.stage:
            raise NeuralPathConfigError("stage 不能为空")
        if not self.baseline_potential:
            raise NeuralPathConfigError("baseline_potential 不能为空")
        if not math.isfinite(self.endpoint_tolerance) or self.endpoint_tolerance < 0:
            raise NeuralPathConfigError("endpoint_tolerance 必须是有限非负数")
        if (
            not math.isfinite(self.max_abs_coefficient)
            or self.max_abs_coefficient < 0
        ):
            raise NeuralPathConfigError("max_abs_coefficient 必须是有限非负数")
        if not self.coefficients:
            raise NeuralPathConfigError("coefficients 不能为空")
        if any(not math.isfinite(value) for value in self.coefficients):
            raise NeuralPathConfigError("coefficients 必须全部有限")
        if any(abs(value) > self.max_abs_coefficient for value in self.coefficients):
            raise NeuralPathConfigError(
                "存在 coefficient 超过 max_abs_coefficient"
            )
        if self.enabled:
            if len(self.coefficients) != 1:
                raise NeuralPathConfigError("协议 v1 启用时严格要求 M=1")
            if len(self.bases) != len(self.coefficients):
                raise NeuralPathConfigError(
                    "启用时 bases 数量必须与 coefficients 数量一致"
                )
            if self.safety is None:
                raise NeuralPathConfigError("启用时必须提供 safety 配置")
        elif self.bases and len(self.bases) != len(self.coefficients):
            raise NeuralPathConfigError(
                "若禁用配置仍声明 bases，其数量必须与 coefficients 一致"
            )

        # 构造时即验证严格端点，而不是等到 production 才发现路径不闭合。
        for endpoint in (0.0, 1.0):
            if abs(self.envelope(endpoint)) > self.endpoint_tolerance:
                raise NeuralPathConfigError(
                    f"包络在 λ={endpoint:g} 未在端点容差内归零"
                )

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
        *,
        verify_basis_files: bool = True,
    ) -> "OuterLambdaController":
        """从完整配置或 ``neural_path`` 子映射构造控制器。"""

        if not isinstance(config, Mapping):
            raise NeuralPathConfigError("配置必须是映射")
        raw = config.get("neural_path", config)
        if not isinstance(raw, Mapping):
            raise NeuralPathConfigError("neural_path 必须是映射")

        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise NeuralPathConfigError("neural_path.enabled 必须是 bool")

        protocol_version = raw.get(
            "protocol_version", NEURAL_PATH_PROTOCOL_VERSION
        )
        if isinstance(protocol_version, bool) or not isinstance(protocol_version, int):
            raise NeuralPathConfigError("neural_path.protocol_version 必须是整数")

        envelope = raw.get("envelope", {"type": "sin2", "parameters": {}})
        if not isinstance(envelope, Mapping):
            raise NeuralPathConfigError("neural_path.envelope 必须是映射")
        envelope_type = _nonempty_string(
            envelope.get("type"), "neural_path.envelope.type"
        )
        parameters = envelope.get("parameters", {})
        if not isinstance(parameters, Mapping) or parameters:
            raise NeuralPathConfigError(
                "协议 v1 的 sin2 envelope.parameters 必须是空映射"
            )

        coefficient_model = raw.get("coefficient_model")
        if not isinstance(coefficient_model, Mapping):
            raise NeuralPathConfigError("neural_path.coefficient_model 必须是映射")
        coefficient_type = _nonempty_string(
            coefficient_model.get("type"),
            "neural_path.coefficient_model.type",
        )
        raw_coefficients = coefficient_model.get("coefficients")
        if (
            not isinstance(raw_coefficients, Sequence)
            or isinstance(raw_coefficients, (str, bytes))
            or not raw_coefficients
        ):
            raise NeuralPathConfigError(
                "neural_path.coefficient_model.coefficients 必须是非空序列"
            )
        coefficients = tuple(
            _finite_float(value, f"coefficient[{index}]")
            for index, value in enumerate(raw_coefficients)
        )
        max_abs = _finite_float(
            coefficient_model.get("max_abs_coefficient"),
            "neural_path.coefficient_model.max_abs_coefficient",
        )

        raw_bases = raw.get("bases", [])
        if (
            not isinstance(raw_bases, Sequence)
            or isinstance(raw_bases, (str, bytes))
        ):
            raise NeuralPathConfigError("neural_path.bases 必须是序列")
        bases = tuple(
            NeuralBasisModelSpec.from_mapping(
                basis, verify_files=verify_basis_files
            )
            for basis in raw_bases
        )

        raw_safety = raw.get("safety")
        safety = (
            NeuralPathSafety.from_mapping(raw_safety)
            if raw_safety is not None
            else None
        )

        return cls(
            enabled=enabled,
            protocol_version=protocol_version,
            stage=_nonempty_string(raw.get("stage"), "neural_path.stage"),
            baseline_potential=_nonempty_string(
                raw.get("baseline_potential"),
                "neural_path.baseline_potential",
            ),
            endpoint_tolerance=_finite_float(
                raw.get("endpoint_tolerance", 1.0e-12),
                "neural_path.endpoint_tolerance",
            ),
            envelope_type=envelope_type,
            coefficient_model_type=coefficient_type,
            coefficients=coefficients,
            max_abs_coefficient=max_abs,
            bases=bases,
            safety=safety,
        )

    @property
    def basis_count(self) -> int:
        return len(self.coefficients)

    def envelope(self, lambda_value: float) -> float:
        """计算 ``w(lambda)``；端点显式返回 +0.0，保证逐位稳定。"""

        lam = _finite_float(lambda_value, "lambda")
        if lam < 0.0 or lam > 1.0:
            raise NeuralPathConfigError(f"lambda 必须位于 [0, 1]，收到 {lam!r}")
        if lam == 0.0 or lam == 1.0:
            return 0.0
        return math.sin(math.pi * lam) ** 2

    def coefficient_vector(self, lambda_value: float) -> tuple[float, ...]:
        """返回 ``c_m(lambda)``；协议 v1 中它与 λ 无关。"""

        # 即使常数模型也验证 λ，防调用方把非法状态悄悄写入协议。
        self.envelope(lambda_value)
        return self.coefficients

    def state_coefficients(self, lambda_value: float) -> tuple[float, ...]:
        """返回矩阵 A 的一行 ``A_m = w(lambda)c_m(lambda)``。"""

        if not self.enabled:
            return tuple(0.0 for _ in self.coefficients)
        weight = self.envelope(lambda_value)
        if weight == 0.0:
            return tuple(0.0 for _ in self.coefficients)
        return tuple(weight * coefficient for coefficient in self.coefficients)

    def coefficient_matrix(
        self, lambdas: Iterable[float]
    ) -> tuple[tuple[float, ...], ...]:
        """生成不可变的全局 ``A[k][m]`` 系数矩阵。"""

        values = tuple(_finite_float(value, "lambda") for value in lambdas)
        if not values:
            raise NeuralPathConfigError("lambda schedule 不能为空")
        return tuple(self.state_coefficients(value) for value in values)

    def validate_cv_budget(self, state_count: int) -> int:
        """验证文档约束 ``2K + M <= 32``，返回所需 CV 数。"""

        if isinstance(state_count, bool) or not isinstance(state_count, int):
            raise NeuralPathConfigError("state_count 必须是整数")
        if state_count <= 0:
            raise NeuralPathConfigError("state_count 必须大于 0")
        used = 2 * state_count + (self.basis_count if self.enabled else 0)
        if used > _MAX_CUSTOM_CV_VARIABLES:
            raise NeuralPathConfigError(
                f"CustomCVForce 需要 {used} 个 CV，超过上限 "
                f"{_MAX_CUSTOM_CV_VARIABLES} (2K+M)"
            )
        return used

    def protocol_payload(
        self, *, lambdas: Iterable[float] | None = None
    ) -> dict[str, Any]:
        """返回稳定、可 JSON 序列化且足以使旧缓存失效的协议 payload。"""

        payload: dict[str, Any] = {
            "protocol_version": self.protocol_version,
            "model_protocol_version": NEURAL_BASIS_MODEL_PROTOCOL_VERSION,
            "accounting_version": NEURAL_PATH_ACCOUNTING_VERSION,
            "enabled": self.enabled,
            "stage": self.stage,
            "baseline_potential": self.baseline_potential,
            "endpoint_tolerance": self.endpoint_tolerance,
            "envelope": {"type": self.envelope_type, "parameters": {}},
            "coefficient_model": {
                "type": self.coefficient_model_type,
                "coefficients": list(self.coefficients),
                "max_abs_coefficient": self.max_abs_coefficient,
            },
            "bases": [basis.protocol_payload() for basis in self.bases],
            "safety": self.safety.protocol_payload() if self.safety else None,
        }
        if lambdas is not None:
            schedule = tuple(_finite_float(value, "lambda") for value in lambdas)
            payload["lambda_schedule"] = list(schedule)
            payload["coefficient_matrix"] = [
                list(row) for row in self.coefficient_matrix(schedule)
            ]
        # 立即做一次规范序列化，确保返回值不存在 NaN 或不稳定类型。
        _canonical_json_bytes(payload)
        return payload

    def protocol_sha256(self, *, lambdas: Iterable[float] | None = None) -> str:
        return stable_payload_sha256(self.protocol_payload(lambdas=lambdas))

    def centered_basis_energies(
        self, basis_energies_kj_mol: Sequence[float]
    ) -> tuple[float, ...]:
        """计算 ``U_m-b_m`` 并执行有限性/安全阈值 hard gate。"""

        if len(basis_energies_kj_mol) != self.basis_count:
            raise NeuralPathConfigError(
                f"期望 {self.basis_count} 个基势能量，"
                f"收到 {len(basis_energies_kj_mol)} 个"
            )
        if self.bases and len(self.bases) != self.basis_count:
            raise NeuralPathConfigError("bases 与 coefficients 数量不一致")

        centered = []
        for index, raw_energy in enumerate(basis_energies_kj_mol):
            energy = _finite_float(raw_energy, f"basis_energy[{index}]")
            offset = (
                self.bases[index].energy_offset_kj_mol if self.bases else 0.0
            )
            value = energy - offset
            if (
                self.enabled
                and self.safety is not None
                and abs(value) > self.safety.max_abs_basis_energy_kj_mol
            ):
                raise NeuralPathConfigError(
                    f"centered basis_energy[{index}]={value!r} kJ/mol 超过安全上限 "
                    f"{self.safety.max_abs_basis_energy_kj_mol!r}"
                )
            centered.append(value)
        return tuple(centered)

    def neural_path_state_energies(
        self,
        lambdas: Iterable[float],
        basis_energies_kj_mol: Sequence[float],
    ) -> tuple[float, ...]:
        """同一坐标上只消费 M 个共享基势值，组合出 K 个路径能量。"""

        matrix = self.coefficient_matrix(lambdas)
        if not self.enabled:
            return tuple(0.0 for _ in matrix)
        centered = self.centered_basis_energies(basis_energies_kj_mol)
        energies = []
        for state_index, row in enumerate(matrix):
            value = math.fsum(
                coefficient * energy
                for coefficient, energy in zip(row, centered, strict=True)
            )
            if not math.isfinite(value):
                raise NeuralPathConfigError(
                    f"neural_path_state_energy[{state_index}] 非有限"
                )
            if (
                self.safety is not None
                and abs(value) > self.safety.max_abs_path_energy_kj_mol
            ):
                raise NeuralPathConfigError(
                    f"neural_path_state_energy[{state_index}]={value!r} kJ/mol "
                    f"超过安全上限 {self.safety.max_abs_path_energy_kj_mol!r}"
                )
            energies.append(value)
        return tuple(energies)

    def compose_target_state_energies(
        self,
        original_state_energies_kj_mol: Sequence[float],
        lambdas: Iterable[float],
        basis_energies_kj_mol: Sequence[float],
    ) -> tuple[float, ...]:
        """把路径项加入 target energies；不接收也不修改 sampling bias。

        该 API 的形状有意防止把神经路径项误写入 ``bias_history``。调用方应将返回值
        写入 target ``energy_history``，IBS/WCA bias 继续沿原账本独立保存。
        """

        original = tuple(
            _finite_float(value, f"original_state_energy[{index}]")
            for index, value in enumerate(original_state_energies_kj_mol)
        )
        path = self.neural_path_state_energies(
            lambdas, basis_energies_kj_mol
        )
        if len(original) != len(path):
            raise NeuralPathConfigError(
                f"original state 数量 {len(original)} 与 lambda 数量 "
                f"{len(path)} 不一致"
            )
        target = tuple(
            base + addition
            for base, addition in zip(original, path, strict=True)
        )
        if any(not math.isfinite(value) for value in target):
            raise NeuralPathConfigError("完整 target state energy 出现非有限值")
        return target

    def neural_path_forces(
        self,
        lambda_value: float,
        basis_forces_kj_mol_nm: Sequence[Sequence[Sequence[float]]],
    ) -> tuple[tuple[float, float, float], ...]:
        """组合一个 λ 状态的附加力 ``sum_m A_m F_m``。

        输入形状为 ``[M][N_atoms][3]``。能量基准 ``b_m`` 是常数，因此不改变力。
        启用路径时同时执行每个 basis force 与组合后 path force 的安全阈值门。
        """

        if not isinstance(basis_forces_kj_mol_nm, Sequence):
            raise NeuralPathConfigError("basis_forces 必须是 [M][N_atoms][3] 序列")
        if len(basis_forces_kj_mol_nm) != self.basis_count:
            raise NeuralPathConfigError(
                f"期望 {self.basis_count} 组 basis forces，"
                f"收到 {len(basis_forces_kj_mol_nm)} 组"
            )
        if not basis_forces_kj_mol_nm:
            return ()

        atom_count = len(basis_forces_kj_mol_nm[0])
        normalized: list[tuple[tuple[float, float, float], ...]] = []
        for basis_index, basis_forces in enumerate(basis_forces_kj_mol_nm):
            if len(basis_forces) != atom_count:
                raise NeuralPathConfigError("所有 basis force 的原子数必须一致")
            normalized_basis = []
            for atom_index, vector in enumerate(basis_forces):
                if (
                    not isinstance(vector, Sequence)
                    or isinstance(vector, (str, bytes))
                    or len(vector) != 3
                ):
                    raise NeuralPathConfigError(
                        f"basis_force[{basis_index}][{atom_index}] 必须是三维向量"
                    )
                xyz = tuple(
                    _finite_float(
                        component,
                        f"basis_force[{basis_index}][{atom_index}][{axis}]",
                    )
                    for axis, component in enumerate(vector)
                )
                norm = math.sqrt(math.fsum(component * component for component in xyz))
                if (
                    self.enabled
                    and self.safety is not None
                    and norm > self.safety.max_force_norm_kj_mol_nm
                ):
                    raise NeuralPathConfigError(
                        f"basis_force[{basis_index}][{atom_index}] norm={norm!r} "
                        "kJ/(mol*nm) 超过安全上限 "
                        f"{self.safety.max_force_norm_kj_mol_nm!r}"
                    )
                normalized_basis.append(xyz)
            normalized.append(tuple(normalized_basis))

        row = self.state_coefficients(lambda_value)
        combined = []
        for atom_index in range(atom_count):
            vector = tuple(
                math.fsum(
                    row[basis_index] * normalized[basis_index][atom_index][axis]
                    for basis_index in range(self.basis_count)
                )
                for axis in range(3)
            )
            norm = math.sqrt(math.fsum(component * component for component in vector))
            if (
                self.enabled
                and self.safety is not None
                and norm > self.safety.max_force_norm_kj_mol_nm
            ):
                raise NeuralPathConfigError(
                    f"path_force[{atom_index}] norm={norm!r} kJ/(mol*nm) "
                    f"超过安全上限 {self.safety.max_force_norm_kj_mol_nm!r}"
                )
            combined.append(vector)
        return tuple(combined)

    def evaluate_support_domains(
        self,
        positions_nm: Sequence[Sequence[float]],
        box_vectors_nm: Sequence[Sequence[float]] | None = None,
    ) -> tuple[SupportDomainEvaluation, ...]:
        """评价固定选择的几何支持域；周期 basis 使用 triclinic minimum image。"""

        normalized = []
        for atom_index, position in enumerate(positions_nm):
            if (
                not isinstance(position, Sequence)
                or isinstance(position, (str, bytes))
                or len(position) != 3
            ):
                raise NeuralPathConfigError(
                    f"position[{atom_index}] 必须是三维坐标"
                )
            normalized.append(
                tuple(
                    _finite_float(
                        component, f"position[{atom_index}][{axis}]"
                    )
                    for axis, component in enumerate(position)
                )
            )
        if not normalized:
            raise NeuralPathConfigError("support-domain positions 不能为空")
        box_rows = _normalize_box_vectors_nm(box_vectors_nm)
        inverse_box_rows = (
            _inverse_3x3_rows(box_rows) if box_rows is not None else None
        )

        evaluations = []
        for basis in self.bases:
            indices = basis.atom_indices()
            if max(indices) >= len(normalized):
                raise NeuralPathConfigError(
                    f"basis[{basis.name}] 原子选择最大索引 {max(indices)} "
                    f"超出坐标原子数 {len(normalized)}"
                )
            selected = [normalized[index] for index in indices]
            uses_minimum_image = (
                basis.periodic
                or basis.coordinate_imaging == "minimum_image_local"
            )
            if uses_minimum_image and box_rows is None:
                raise NeuralPathConfigError(
                    f"basis[{basis.name}] 的支持域要求 minimum-image，"
                    "必须提供 box_vectors_nm"
                )
            if uses_minimum_image and box_rows is not None:
                anchor = selected[0]
                unwrapped = [anchor]
                for position in selected[1:]:
                    displacement = tuple(
                        position[axis] - anchor[axis] for axis in range(3)
                    )
                    minimum_image = _minimum_image_displacement(
                        displacement, box_rows, inverse_box_rows
                    )
                    unwrapped.append(
                        tuple(
                            anchor[axis] + minimum_image[axis]
                            for axis in range(3)
                        )
                    )
                selected_for_shape = unwrapped
            else:
                selected_for_shape = selected
            center = tuple(
                statistics.fmean(
                    position[axis] for position in selected_for_shape
                )
                for axis in range(3)
            )
            radius_of_gyration = math.sqrt(
                math.fsum(
                    math.fsum(
                        (position[axis] - center[axis]) ** 2
                        for axis in range(3)
                    )
                    for position in selected_for_shape
                )
                / len(selected_for_shape)
            )
            pair_distances = []
            for left in range(len(selected)):
                for right in range(left + 1, len(selected)):
                    displacement = tuple(
                        selected[left][axis] - selected[right][axis]
                        for axis in range(3)
                    )
                    if uses_minimum_image and box_rows is not None:
                        displacement = _minimum_image_displacement(
                            displacement, box_rows, inverse_box_rows
                        )
                    pair_distances.append(
                        math.sqrt(
                            math.fsum(
                                component * component
                                for component in displacement
                            )
                        )
                    )
            min_pair = min(pair_distances) if pair_distances else None
            max_pair = max(pair_distances) if pair_distances else None
            violations = []
            domain = basis.support_domain
            if domain is not None:
                if domain.min_pair_distance_nm is not None:
                    if min_pair is None:
                        violations.append(
                            "min_pair_distance_requires_at_least_two_atoms"
                        )
                    elif min_pair < domain.min_pair_distance_nm:
                        violations.append("min_pair_distance_below_support")
                if domain.max_pair_distance_nm is not None:
                    if max_pair is None:
                        violations.append(
                            "max_pair_distance_requires_at_least_two_atoms"
                        )
                    elif max_pair > domain.max_pair_distance_nm:
                        violations.append("max_pair_distance_above_support")
                if (
                    domain.max_radius_of_gyration_nm is not None
                    and radius_of_gyration
                    > domain.max_radius_of_gyration_nm
                ):
                    violations.append("radius_of_gyration_above_support")
            evaluations.append(
                SupportDomainEvaluation(
                    basis_name=basis.name,
                    supported=not violations,
                    selected_atom_count=len(selected),
                    min_pair_distance_nm=min_pair,
                    max_pair_distance_nm=max_pair,
                    radius_of_gyration_nm=radius_of_gyration,
                    violations=tuple(violations),
                )
            )
        return tuple(evaluations)


@dataclass(frozen=True)
class AnalyticBasisEvaluation:
    """解析 mock 基势在一帧坐标上的能量和守恒力。"""

    energy_kj_mol: float
    forces_kj_mol_nm: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class HarmonicDistanceBasis:
    """两个固定原子间的平移/旋转不变谐波 mock 基势。

    ``U = 0.5*k*(r-r0)^2``。它只用于 WP-2 接口、端点和账本测试，不代表真实神经
    模型。第一版明确不处理 PBC；周期体系测试应先把选中原子成像到同一局部环境。
    """

    atom_i: int
    atom_j: int
    force_constant_kj_mol_nm2: float
    equilibrium_distance_nm: float

    def __post_init__(self) -> None:
        for field_name in ("atom_i", "atom_j"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise NeuralPathConfigError(f"{field_name} 必须是非负整数")
        if self.atom_i == self.atom_j:
            raise NeuralPathConfigError("harmonic basis 的两个原子必须不同")
        if (
            not math.isfinite(self.force_constant_kj_mol_nm2)
            or self.force_constant_kj_mol_nm2 <= 0.0
        ):
            raise NeuralPathConfigError("force_constant 必须是有限正数")
        if (
            not math.isfinite(self.equilibrium_distance_nm)
            or self.equilibrium_distance_nm < 0.0
        ):
            raise NeuralPathConfigError("equilibrium_distance 必须是有限非负数")

    def evaluate(
        self, positions_nm: Sequence[Sequence[float]]
    ) -> AnalyticBasisEvaluation:
        """计算标量能量及所有原子的负梯度。"""

        if not isinstance(positions_nm, Sequence) or isinstance(
            positions_nm, (str, bytes)
        ):
            raise NeuralPathConfigError("positions_nm 必须是 [N_atoms][3] 序列")
        atom_count = len(positions_nm)
        if max(self.atom_i, self.atom_j) >= atom_count:
            raise NeuralPathConfigError(
                f"mock basis 原子索引超出 positions 范围 N={atom_count}"
            )

        normalized_positions = []
        for atom_index, position in enumerate(positions_nm):
            if (
                not isinstance(position, Sequence)
                or isinstance(position, (str, bytes))
                or len(position) != 3
            ):
                raise NeuralPathConfigError(
                    f"position[{atom_index}] 必须是三维坐标"
                )
            normalized_positions.append(
                tuple(
                    _finite_float(
                        component, f"position[{atom_index}][{axis}]"
                    )
                    for axis, component in enumerate(position)
                )
            )

        delta = tuple(
            normalized_positions[self.atom_i][axis]
            - normalized_positions[self.atom_j][axis]
            for axis in range(3)
        )
        distance = math.sqrt(math.fsum(value * value for value in delta))
        if distance == 0.0:
            raise NeuralPathConfigError(
                "harmonic distance basis 在 r=0 处方向未定义，拒绝返回伪造力"
            )
        displacement = distance - self.equilibrium_distance_nm
        energy = 0.5 * self.force_constant_kj_mol_nm2 * displacement**2

        # F_i = -dU/dx_i = -k(r-r0) * (x_i-x_j)/r；F_j = -F_i。
        scale = -self.force_constant_kj_mol_nm2 * displacement / distance
        force_i = tuple(scale * value for value in delta)
        force_j = tuple(-value for value in force_i)
        forces = [(0.0, 0.0, 0.0) for _ in range(atom_count)]
        forces[self.atom_i] = force_i
        forces[self.atom_j] = force_j
        return AnalyticBasisEvaluation(
            energy_kj_mol=energy,
            forces_kj_mol_nm=tuple(forces),
        )


@dataclass(frozen=True)
class ExistingOpenMMMLBasisEvaluation:
    """现有 MACE/ORB 局部分解器的一帧能量和映射回全坐标的力。"""

    model_name: str
    label_mode: str
    energy_kj_mol: float
    forces_kj_mol_nm: tuple[tuple[float, float, float], ...]
    max_force_norm_kj_mol_nm: float


class ExistingOrbMaceBasisAdapter:
    """直接复用项目现有 ``Orbv3DEXPFittingPipeline`` 的薄适配器。

    MACE 模型使用现成的 ``E(complex)-E(ligand)-E(environment)``；ORB v3
    模型使用 ``interaction_energy``。Context 缓存、CUDA 回退和清理都继续由
    原类负责。
    """

    def __init__(
        self,
        model_name: str = "mace-off24-medium",
        device: str = "cuda",
    ) -> None:
        self.model_name = _nonempty_string(model_name, "model_name")
        requested_device = _nonempty_string(device, "device").lower()
        if requested_device not in {"cpu", "cuda"}:
            raise NeuralPathConfigError("device 必须是 'cpu' 或 'cuda'")
        try:
            from dexp_退役 import Orbv3DEXPFittingPipeline
        except Exception as exc:
            raise TorchForceDeploymentError(
                "无法导入现有 dexp_退役.Orbv3DEXPFittingPipeline"
            ) from exc
        try:
            self._pipeline = Orbv3DEXPFittingPipeline(
                model_name=self.model_name,
                device=requested_device,
            )
        except Exception as exc:
            raise TorchForceDeploymentError(
                f"现有 MACE/ORB pipeline 初始化失败 ({self.model_name}): {exc}"
            ) from exc
        self.device = str(self._pipeline.device)
        self._source_label_mode = str(self._pipeline.label_mode)
        if self._source_label_mode == "orbv3_interaction":
            if "conservative" not in self.model_name.lower():
                self._pipeline._clear_orb_context_cache()
                raise NeuralPathConfigError(
                    "外层路径只接受 conservative ORB 模型"
                )
            # 当前 OpenMM-ML OrbPotential 忽略 returnEnergyType；不能把局部总能量
            # 误标成 interaction energy，因此显式做三体能量/力分解。
            self.label_mode = "orbv3_decomposition"
        else:
            self.label_mode = self._source_label_mode
        self._closed = False

    def _get_decomposition_bundle(
        self,
        ligand_array,
        environment_array,
        number_array,
    ):
        import numpy as np

        if self._source_label_mode != "orbv3_interaction":
            return self._pipeline._get_orb_decomposition_bundle(
                ligand_array,
                environment_array,
                number_array,
            )
        # 🔑 [2026-09-09] key 里补上原子序数。
        #
        # 原来 key 只有 (标签, ligand 索引, environment 索引, device)，**不含
        # `number_array`** —— 同一组索引配不同的原子序数会静默命中同一份缓存，
        # 拿到为别的元素组成建的 ORB/MACE Context，算出一个不报错的错数。
        key = (
            "outer_lambda_orbv3_decomposition",
            tuple(int(value) for value in ligand_array),
            tuple(int(value) for value in environment_array),
            tuple(int(value) for value in number_array),
            self.device,
        )
        cache = self._pipeline._orb_ctx_cache
        if key not in cache:
            # 🔑 [2026-09-09] 给这个缓存加上限。
            #
            # 每个 miss 会建 **3 个** OpenMM Context，每个都带一个 GPU 上的
            # ORB/MACE 模型，而这里原来既无淘汰也无容量上限。只要调用方在帧之间
            # 变动 environment 选择（按距离选壳、逐窗口选择），就会无界增长到
            # 显存耗尽；今天没炸只是因为现有调用方全程用同一份冻结选择。
            # 用最简单的 FIFO：超了就把最早的一条连同它那 3 个 Context 一起丢。
            while len(cache) >= _ORB_DECOMPOSITION_CACHE_MAX_ENTRIES:
                evicted_key = next(iter(cache))
                cache.pop(evicted_key, None)
            combined = np.concatenate(
                [ligand_array, environment_array]
            )
            cache[key] = {
                "comb_idx": combined,
                "lig_idx": ligand_array,
                "env_idx": environment_array,
                "contexts": {
                    "cplx": self._pipeline._create_orb_context_bundle(
                        number_array[combined]
                    ),
                    "lig": self._pipeline._create_orb_context_bundle(
                        number_array[ligand_array]
                    ),
                    "env": self._pipeline._create_orb_context_bundle(
                        number_array[environment_array]
                    ),
                },
            }
        return cache[key]

    @staticmethod
    def _normalize_indices(
        values: Sequence[int], field: str, atom_count: int
    ) -> tuple[int, ...]:
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or not values
        ):
            raise NeuralPathConfigError(f"{field} 必须是非空整数序列")
        normalized = []
        for position, value in enumerate(values):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= atom_count
            ):
                raise NeuralPathConfigError(
                    f"{field}[{position}]={value!r} 超出原子范围"
                )
            normalized.append(value)
        if len(set(normalized)) != len(normalized):
            raise NeuralPathConfigError(f"{field} 不允许重复")
        return tuple(normalized)

    def evaluate(
        self,
        positions_nm: Sequence[Sequence[float]],
        *,
        ligand_indices: Sequence[int],
        environment_indices: Sequence[int],
        atomic_numbers: Sequence[int],
    ) -> ExistingOpenMMMLBasisEvaluation:
        if self._closed:
            raise TorchForceDeploymentError("MACE/ORB adapter 已关闭")
        frame = _normalize_frame_collection([positions_nm])[0]
        atom_count = len(frame)
        ligand = self._normalize_indices(
            ligand_indices, "ligand_indices", atom_count
        )
        environment = self._normalize_indices(
            environment_indices, "environment_indices", atom_count
        )
        overlap = set(ligand).intersection(environment)
        if overlap:
            raise NeuralPathConfigError(
                f"ligand/environment 原子集合重叠: {sorted(overlap)}"
            )
        if (
            not isinstance(atomic_numbers, Sequence)
            or isinstance(atomic_numbers, (str, bytes))
            or len(atomic_numbers) != atom_count
        ):
            raise NeuralPathConfigError(
                "atomic_numbers 必须与 positions 原子数一致"
            )
        numbers = []
        for index, value in enumerate(atomic_numbers):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise NeuralPathConfigError(
                    f"atomic_numbers[{index}] 必须是正整数"
                )
            numbers.append(value)

        try:
            import numpy as np
            from openmm import unit

            position_array = np.asarray(frame, dtype=np.float64)
            ligand_array = np.asarray(ligand, dtype=int)
            environment_array = np.asarray(environment, dtype=int)
            number_array = np.asarray(numbers, dtype=int)
            if self._source_label_mode != "orbv3_interaction":
                self._pipeline._preflight_orb_backend(
                    position_array,
                    ligand_array,
                    environment_array,
                    number_array,
                )
            bundle = self._get_decomposition_bundle(
                ligand_array,
                environment_array,
                number_array,
            )

            def state_energy_forces(label: str, indices):
                context = bundle["contexts"][label]["context"]
                context.setPositions(position_array[indices] * unit.nanometer)
                state = context.getState(getEnergy=True, getForces=True)
                energy = state.getPotentialEnergy().value_in_unit(
                    unit.kilojoules_per_mole
                )
                forces = state.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoules_per_mole / unit.nanometer
                )
                return float(energy), np.asarray(forces, dtype=np.float64)

            energy, combined_forces = state_energy_forces(
                "cplx", bundle["comb_idx"]
            )
            full_forces = np.zeros((atom_count, 3), dtype=np.float64)
            full_forces[bundle["comb_idx"]] += combined_forces
            if self.label_mode in {
                "mace_decomposition",
                "orbv3_decomposition",
            }:
                ligand_energy, ligand_forces = state_energy_forces(
                    "lig", bundle["lig_idx"]
                )
                environment_energy, environment_forces = (
                    state_energy_forces("env", bundle["env_idx"])
                )
                energy -= ligand_energy + environment_energy
                full_forces[bundle["lig_idx"]] -= ligand_forces
                full_forces[bundle["env_idx"]] -= environment_forces
        except Exception as exc:
            if isinstance(
                exc,
                (
                    NeuralPathConfigError,
                    NeuralPathIntegrityError,
                    TorchForceDeploymentError,
                ),
            ):
                raise
            raise TorchForceDeploymentError(
                f"{self.model_name} 局部能量/力评价失败: {exc}"
            ) from exc

        energy = _finite_float(energy, "existing OpenMM-ML basis energy")
        force_payload = tuple(
            tuple(
                _finite_float(value, "existing OpenMM-ML basis force")
                for value in vector
            )
            for vector in full_forces
        )
        norms = tuple(
            math.sqrt(math.fsum(component * component for component in vector))
            for vector in force_payload
        )
        return ExistingOpenMMMLBasisEvaluation(
            model_name=self.model_name,
            label_mode=self.label_mode,
            energy_kj_mol=energy,
            forces_kj_mol_nm=force_payload,
            max_force_norm_kj_mol_nm=max(norms, default=0.0),
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._pipeline._clear_orb_context_cache()
        finally:
            self._closed = True

    def __enter__(self) -> "ExistingOrbMaceBasisAdapter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class MaceDecompositionPythonComputation:
    """单模型三次前向的 MACE 局部分解 PythonForce callback。

    这是 WP-4 的每步执行桥接后端：模型只加载一次，固定选择在每步按 triclinic
    minimum image 成像，然后依次计算 complex/ligand/environment。它比三个常驻
    OpenMM probe Context 更省模型内存，但仍有三次 MACE 前向；后续可替换为 batch
    实现而不改变外层协议。
    """

    def __init__(
        self,
        *,
        model_path: str,
        model_sha256: str,
        atomic_numbers: Sequence[int],
        ligand_indices: Sequence[int],
        environment_indices: Sequence[int],
        coefficient: float,
        energy_offset_kj_mol: float,
        lambda_parameter_name: str,
        device: str,
        precision: str,
        max_abs_basis_energy_kj_mol: float,
        max_abs_path_energy_kj_mol: float,
        max_force_norm_kj_mol_nm: float,
    ) -> None:
        self.model_path = str(
            _absolute_file_path(model_path, "model_path")
        )
        self.model_sha256 = _normalize_sha256(
            model_sha256, "model_sha256"
        )
        self.atomic_numbers = tuple(atomic_numbers)
        if not self.atomic_numbers or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in self.atomic_numbers
        ):
            raise NeuralPathConfigError(
                "atomic_numbers 必须是非空正整数序列"
            )
        atom_count = len(self.atomic_numbers)
        self.ligand_indices = ExistingOrbMaceBasisAdapter._normalize_indices(
            ligand_indices, "ligand_indices", atom_count
        )
        self.environment_indices = (
            ExistingOrbMaceBasisAdapter._normalize_indices(
                environment_indices, "environment_indices", atom_count
            )
        )
        if set(self.ligand_indices).intersection(self.environment_indices):
            raise NeuralPathConfigError(
                "ligand_indices 与 environment_indices 不允许重叠"
            )
        self.combined_indices = (
            self.ligand_indices + self.environment_indices
        )
        self.coefficient = _finite_float(coefficient, "coefficient")
        self.energy_offset_kj_mol = _finite_float(
            energy_offset_kj_mol, "energy_offset_kj_mol"
        )
        self.lambda_parameter_name = _nonempty_string(
            lambda_parameter_name, "lambda_parameter_name"
        )
        self.device = _nonempty_string(device, "device").lower()
        if self.device not in {"cpu", "cuda"}:
            raise NeuralPathConfigError("device 必须是 cpu 或 cuda")
        self.precision = _nonempty_string(precision, "precision")
        if self.precision not in {"single", "double"}:
            raise NeuralPathConfigError("precision 必须是 single 或 double")
        self.max_abs_basis_energy_kj_mol = _finite_float(
            max_abs_basis_energy_kj_mol,
            "max_abs_basis_energy_kj_mol",
        )
        self.max_abs_path_energy_kj_mol = _finite_float(
            max_abs_path_energy_kj_mol,
            "max_abs_path_energy_kj_mol",
        )
        self.max_force_norm_kj_mol_nm = _finite_float(
            max_force_norm_kj_mol_nm,
            "max_force_norm_kj_mol_nm",
        )
        if min(
            self.max_abs_basis_energy_kj_mol,
            self.max_abs_path_energy_kj_mol,
            self.max_force_norm_kj_mol_nm,
        ) <= 0.0:
            raise NeuralPathConfigError("MACE PythonForce safety limits 必须为正")
        self._model = None
        self._torch = None
        self._region_static = None

    def __getstate__(self):
        state = dict(self.__dict__)
        # XML/pickle 只保存重建规格，不嵌入运行时模型/GPU tensors。
        state["_model"] = None
        state["_torch"] = None
        state["_region_static"] = None
        return state

    def _load_model(self) -> None:
        if self._model is not None:
            return
        actual_sha = sha256_file(self.model_path)
        if actual_sha != self.model_sha256:
            raise NeuralPathIntegrityError(
                "MACE PythonForce 加载时模型 SHA-256 已变化"
            )
        try:
            import torch
            from mace.tools import (
                AtomicNumberTable,
                atomic_numbers_to_indices,
                to_one_hot,
            )
        except ImportError as exc:
            raise TorchForceDeploymentError(
                "MACE PythonForce 需要 torch 和 mace"
            ) from exc
        if self.device == "cuda" and not torch.cuda.is_available():
            raise TorchForceDeploymentError(
                "请求 CUDA MACE PythonForce，但 torch.cuda 不可用"
            )
        torch_device = torch.device(self.device)
        try:
            model = torch.load(
                self.model_path,
                map_location=torch_device,
                weights_only=False,
            ).to(torch_device)
        except Exception as exc:
            raise TorchForceDeploymentError(
                f"无法加载 MACE 模型 {self.model_path}: {exc}"
            ) from exc
        dtype = torch.float32 if self.precision == "single" else torch.float64
        model = model.to(dtype=dtype)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        z_table = AtomicNumberTable(
            [int(value) for value in model.atomic_numbers]
        )
        region_static = {}
        for label, indices in (
            ("cplx", self.combined_indices),
            ("lig", self.ligand_indices),
            ("env", self.environment_indices),
        ):
            numbers = [self.atomic_numbers[index] for index in indices]
            try:
                encoded = atomic_numbers_to_indices(
                    numbers, z_table=z_table
                )
            except Exception as exc:
                raise NeuralPathConfigError(
                    f"MACE 模型不覆盖 {label} 区域中的全部元素: {exc}"
                ) from exc
            node_attrs = to_one_hot(
                torch.tensor(
                    encoded,
                    dtype=torch.long,
                    device=torch_device,
                ).unsqueeze(-1),
                num_classes=len(z_table),
            ).to(dtype)
            count = len(indices)
            region_static[label] = {
                "indices": indices,
                "ptr": torch.tensor(
                    [0, count],
                    dtype=torch.long,
                    device=torch_device,
                ),
                "node_attrs": node_attrs,
                "batch": torch.zeros(
                    count, dtype=torch.long, device=torch_device
                ),
                "pbc": torch.tensor(
                    [False, False, False],
                    dtype=torch.bool,
                    device=torch_device,
                ),
                "charge": torch.tensor(
                    [0.0], dtype=dtype, device=torch_device
                ),
                "multiplicity": torch.tensor(
                    [1.0], dtype=dtype, device=torch_device
                ),
            }
        self._model = model
        self._torch = torch
        self._region_static = region_static

    @staticmethod
    def _minimum_image_selected(positions_nm, indices, box_nm):
        import numpy as np

        selected = np.asarray(positions_nm[indices], dtype=np.float64)
        anchor = selected[0].copy()
        inverse_box = np.linalg.inv(box_nm)
        displacement = selected - anchor
        fractional = displacement @ inverse_box
        displacement -= np.floor(fractional + 0.5) @ box_nm
        return anchor + displacement

    def _evaluate_region(self, label: str, positions_nm):
        import numpy as np
        from mace.data.neighborhood import get_neighborhood

        static = self._region_static[label]
        positions_angstrom = np.asarray(positions_nm, dtype=np.float64) * 10.0
        cell = np.identity(3, dtype=np.float64)
        cutoff = float(self._model.r_max.detach())
        edge_index, shifts, _, _ = get_neighborhood(
            positions_angstrom,
            cutoff,
            [False, False, False],
            cell,
        )
        torch = self._torch
        dtype = static["node_attrs"].dtype
        device = static["ptr"].device
        inputs = {
            "ptr": static["ptr"],
            "node_attrs": static["node_attrs"],
            "batch": static["batch"],
            "pbc": static["pbc"],
            "positions": torch.tensor(
                positions_angstrom, dtype=dtype, device=device
            ),
            "edge_index": torch.tensor(
                edge_index, dtype=torch.int64, device=device
            ),
            "shifts": torch.tensor(
                shifts, dtype=dtype, device=device
            ),
            "cell": torch.tensor(cell, dtype=dtype, device=device),
            "total_charge": static["charge"],
            "total_spin": static["multiplicity"],
        }
        results = self._model(inputs, compute_force=True)
        energy_ev = float(results["interaction_energy"].detach())
        forces_ev_angstrom = (
            results["forces"].detach().cpu().numpy().astype(np.float64)
        )
        return (
            energy_ev * 96.4853,
            forces_ev_angstrom * 96.4853 * 10.0,
        )

    def _evaluate_decomposition(self, positions_nm, box_nm):
        import numpy as np

        self._load_model()
        # 🔑 [2026-09-10] 三个区域**共用同一次 minimum image**，再切片。
        #
        # 原来是每个区域各做一次 `_minimum_image_selected`，而那个函数把
        # `selected[0]` 当锚点 —— cplx 和 lig 锚在 `ligand_indices[0]`，env 却锚在
        # `environment_indices[0]`。离锚点超过半个盒长的原子会被选到**不同的**
        # 周期镜像，于是 `E_cplx` 里的环境子几何 ≠ 单独算的 `E_env` 的几何，
        # 差值 `E_cplx − E_lig − E_env` 吸收一个伪的胞内项。
        #
        # ⚠️ 关键是**三项用同一套镜像**，不是"镜像本身要物理"：只要三者坐标逐位
        # 同源，这个分解就精确成立，哪怕环境壳大于 L/2、镜像看起来不漂亮。
        # 别"好心地"把 env 按它自己的锚点重新 image 一遍——那正是这个 bug。
        #
        # `combined_indices == ligand_indices + environment_indices`（见 __init__），
        # 所以切片位置是确定的；这里断言一次，防止哪天构造顺序被改。
        n_ligand = len(self.ligand_indices)
        if tuple(self.combined_indices) != tuple(self.ligand_indices) + tuple(
            self.environment_indices
        ):
            raise NeuralPathConfigError(
                "combined_indices 不再是 ligand_indices + environment_indices 的拼接；"
                "下面按位置切片的假设不成立，拒绝继续（会静默给出错误的分解能量）。"
            )
        imaged_combined = self._minimum_image_selected(
            positions_nm, list(self.combined_indices), box_nm
        )
        regions = {}
        for label, imaged in (
            ("cplx", imaged_combined),
            ("lig", imaged_combined[:n_ligand]),
            ("env", imaged_combined[n_ligand:]),
        ):
            regions[label] = self._evaluate_region(label, imaged)
        energy = regions["cplx"][0] - regions["lig"][0] - regions["env"][0]
        full_forces = np.zeros(
            (len(self.atomic_numbers), 3), dtype=np.float64
        )
        full_forces[list(self.combined_indices)] += regions["cplx"][1]
        full_forces[list(self.ligand_indices)] -= regions["lig"][1]
        full_forces[list(self.environment_indices)] -= regions["env"][1]
        return energy, full_forces

    def __call__(self, state):
        import numpy as np

        parameters = state.getParameters()
        if self.lambda_parameter_name not in parameters:
            raise NeuralPathConfigError(
                f"PythonForce State 缺少全局参数 {self.lambda_parameter_name!r}"
            )
        lam = _finite_float(
            parameters[self.lambda_parameter_name],
            self.lambda_parameter_name,
        )
        if lam < 0.0 or lam > 1.0:
            raise NeuralPathConfigError("outer lambda 必须位于 [0,1]")
        atom_count = len(self.atomic_numbers)
        if lam == 0.0 or lam == 1.0:
            return 0.0, np.zeros((atom_count, 3), dtype=np.float64)
        amplitude = self.coefficient * math.sin(math.pi * lam) ** 2
        centered_energy, basis_forces = (
            self._evaluate_centered_basis_from_state(state)
        )
        path_energy = amplitude * centered_energy
        path_forces = amplitude * basis_forces
        if abs(path_energy) > self.max_abs_path_energy_kj_mol:
            raise NeuralPathFrameError(
                "MACE path energy "
                f"{path_energy:.9g} kJ/mol 超过安全门 "
                f"{self.max_abs_path_energy_kj_mol:.9g} kJ/mol "
                f"(lambda={lam:.9g}, centered_basis={centered_energy:.9g} "
                f"kJ/mol, coefficient={self.coefficient:.9g})"
            )
        if (
            path_forces.size
            and float(np.max(np.linalg.norm(path_forces, axis=1)))
            > self.max_force_norm_kj_mol_nm
        ):
            observed_force = float(
                np.max(np.linalg.norm(path_forces, axis=1))
            )
            raise NeuralPathFrameError(
                "MACE path force "
                f"{observed_force:.9g} kJ/(mol*nm) 超过安全门 "
                f"{self.max_force_norm_kj_mol_nm:.9g} kJ/(mol*nm) "
                f"(lambda={lam:.9g}, coefficient={self.coefficient:.9g})"
            )
        return float(path_energy), path_forces

    def _evaluate_centered_basis_from_state(self, state):
        import numpy as np
        from openmm import unit

        positions = state.getPositions(asNumpy=True).value_in_unit(
            unit.nanometer
        )
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
            unit.nanometer
        )
        basis_energy, basis_forces = self._evaluate_decomposition(
            np.asarray(positions, dtype=np.float64),
            np.asarray(box, dtype=np.float64),
        )
        centered_energy = basis_energy - self.energy_offset_kj_mol
        basis_norms = np.linalg.norm(basis_forces, axis=1)
        if (
            not math.isfinite(centered_energy)
            or not np.all(np.isfinite(basis_forces))
        ):
            raise NeuralPathFrameError(
                "MACE PythonForce 出现非有限能量或力"
            )
        if abs(centered_energy) > self.max_abs_basis_energy_kj_mol:
            raise NeuralPathFrameError("MACE basis energy 超过安全门")
        if (
            basis_norms.size
            and float(np.max(basis_norms))
            > self.max_force_norm_kj_mol_nm
        ):
            raise NeuralPathFrameError("MACE basis force 超过安全门")
        return float(centered_energy), basis_forces


class MaceDecompositionBasisPythonComputation(
    MaceDecompositionPythonComputation
):
    """返回未乘外层系数的中心化 MACE basis，供多个 IBS 状态共享。"""

    def __call__(self, state):
        return self._evaluate_centered_basis_from_state(state)


def build_mace_decomposition_python_force(
    controller: OuterLambdaController,
    *,
    atomic_numbers: Sequence[int],
    ligand_indices: Sequence[int],
    environment_indices: Sequence[int],
    lambda_parameter_name: str = "outer_lambda",
    default_lambda: float = 0.0,
    device: str = "cuda",
    force_group: int = 31,
):
    """构建可直接加入 OpenMM System 的每步 MACE 外层 λ PythonForce。"""

    if not controller.enabled or len(controller.bases) != 1:
        raise NeuralPathConfigError("MACE PythonForce 要求 enabled 且 M=1")
    basis = controller.bases[0]
    if basis.backend != "existing_openmmml":
        raise NeuralPathConfigError(
            "MACE PythonForce 要求 backend='existing_openmmml'"
        )
    if basis.model_name is None or "mace" not in basis.model_name.lower():
        raise NeuralPathConfigError("该 builder 只支持 MACE 模型")
    selected = set(ligand_indices).union(environment_indices)
    if selected != set(basis.atom_indices()):
        raise NeuralPathConfigError(
            "MACE PythonForce 选择与配置 fixed atom selection 不一致"
        )
    if controller.safety is None:
        raise NeuralPathConfigError("MACE PythonForce 要求 safety 配置")
    default_lam = _finite_float(default_lambda, "default_lambda")
    controller.envelope(default_lam)
    if (
        isinstance(force_group, bool)
        or not isinstance(force_group, int)
        or force_group < 0
        or force_group > 31
    ):
        raise NeuralPathConfigError("force_group 必须是 [0,31] 整数")
    computation = MaceDecompositionPythonComputation(
        model_path=basis.model_path,
        model_sha256=basis.sha256,
        atomic_numbers=atomic_numbers,
        ligand_indices=ligand_indices,
        environment_indices=environment_indices,
        coefficient=controller.coefficients[0],
        energy_offset_kj_mol=basis.energy_offset_kj_mol,
        lambda_parameter_name=lambda_parameter_name,
        device=device,
        precision=basis.precision,
        max_abs_basis_energy_kj_mol=(
            controller.safety.max_abs_basis_energy_kj_mol
        ),
        max_abs_path_energy_kj_mol=(
            controller.safety.max_abs_path_energy_kj_mol
        ),
        max_force_norm_kj_mol_nm=(
            controller.safety.max_force_norm_kj_mol_nm
        ),
    )
    openmm, _ = _require_openmm()
    force = openmm.PythonForce(
        computation,
        {lambda_parameter_name: default_lam},
    )
    force.setUsesPeriodicBoundaryConditions(True)
    force.setForceGroup(force_group)
    return force


def build_mace_decomposition_basis_python_force(
    controller: OuterLambdaController,
    *,
    atomic_numbers: Sequence[int],
    ligand_indices: Sequence[int],
    environment_indices: Sequence[int],
    device: str = "cuda",
):
    """构建一次评价、由全部 IBS λ 状态共享的中心化 MACE basis Force。"""

    if not controller.enabled or len(controller.bases) != 1:
        raise NeuralPathConfigError("共享 MACE basis Force 要求 enabled 且 M=1")
    basis = controller.bases[0]
    if basis.backend != "existing_openmmml":
        raise NeuralPathConfigError(
            "共享 MACE basis Force 要求 backend='existing_openmmml'"
        )
    if basis.model_name is None or "mace" not in basis.model_name.lower():
        raise NeuralPathConfigError("共享 basis builder 只支持 MACE 模型")
    if (
        set(ligand_indices).union(environment_indices)
        != set(basis.atom_indices())
    ):
        raise NeuralPathConfigError(
            "共享 MACE basis Force 选择与 fixed atom selection 不一致"
        )
    if controller.safety is None:
        raise NeuralPathConfigError("共享 MACE basis Force 要求 safety 配置")
    computation = MaceDecompositionBasisPythonComputation(
        model_path=basis.model_path,
        model_sha256=basis.sha256,
        atomic_numbers=atomic_numbers,
        ligand_indices=ligand_indices,
        environment_indices=environment_indices,
        coefficient=1.0,
        energy_offset_kj_mol=basis.energy_offset_kj_mol,
        lambda_parameter_name="unused_outer_lambda",
        device=device,
        precision=basis.precision,
        max_abs_basis_energy_kj_mol=(
            controller.safety.max_abs_basis_energy_kj_mol
        ),
        max_abs_path_energy_kj_mol=(
            controller.safety.max_abs_path_energy_kj_mol
        ),
        max_force_norm_kj_mol_nm=(
            controller.safety.max_force_norm_kj_mol_nm
        ),
    )
    openmm, _ = _require_openmm()
    force = openmm.PythonForce(computation)
    force.setUsesPeriodicBoundaryConditions(True)
    return force


class OuterLambdaIBSBiasForce:
    """与现有 IBSBiasForce API 兼容的独立共享-basis 偏置力。

    每个 ``neural_basis_m`` 只作为一个 collective variable 加入一次；各状态
    ``X_k`` 通过冻结的 ``A[k,m]`` 组合它，不会为 K 个状态复制 K 份模型。
    """

    def __init__(
        self,
        controller: OuterLambdaController,
        lambdas: Iterable[float],
        temperature_kelvin: Any,
        basis_forces: Sequence[Any],
        *,
        prefix: str = "abfe_neural",
    ) -> None:
        if not isinstance(controller, OuterLambdaController):
            raise TypeError("controller 必须是 OuterLambdaController")
        self.controller = controller
        self.lambdas = tuple(
            _finite_float(value, "lambda") for value in lambdas
        )
        if not self.lambdas:
            raise NeuralPathConfigError("IBS lambda schedule 不能为空")
        self.n_states = len(self.lambdas)
        self.prefix = _nonempty_string(prefix, "prefix")
        self.controller.validate_cv_budget(self.n_states)
        if len(basis_forces) != self.controller.basis_count:
            raise NeuralPathConfigError(
                "basis_forces 数量必须等于 controller.basis_count"
            )
        openmm, unit = _require_openmm()
        try:
            temperature = temperature_kelvin.value_in_unit(unit.kelvin)
        except AttributeError:
            temperature = _finite_float(
                temperature_kelvin, "temperature_kelvin"
            )
        if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
            raise NeuralPathConfigError("temperature_kelvin 必须为有限正数")
        kt = (
            unit.MOLAR_GAS_CONSTANT_R * (float(temperature) * unit.kelvin)
        ).value_in_unit(unit.kilojoules_per_mole)
        beta = 1.0 / kt
        matrix = self.controller.coefficient_matrix(self.lambdas)

        def state_expression(index: int) -> str:
            path_terms = [
                f"({coefficient:.17g})*neural_basis_{basis_index}"
                for basis_index, coefficient in enumerate(matrix[index])
                if coefficient != 0.0
            ]
            expression = f"cv_{index}_int+cv_{index}_rest"
            if path_terms:
                expression += "+" + "+".join(path_terms)
            expression += f"-{self.prefix}_f_{index}"
            return f"({expression})"

        state_expressions = [
            state_expression(index) for index in range(self.n_states)
        ]
        logit_expressions = {
            index: (
                f"(-beta*(({state_expressions[index]})"
                f"-({state_expressions[0]})))"
            )
            for index in range(1, self.n_states)
        }
        pivot = "0.0"
        for index in range(1, self.n_states):
            pivot = f"max({pivot},{logit_expressions[index]})"
        sum_terms = [f"exp(-({pivot}))"] + [
            f"exp({logit_expressions[index]}-({pivot}))"
            for index in range(1, self.n_states)
        ]
        energy_expression = (
            f"{self.prefix}_bias_scale*("
            f"{state_expressions[0]}-kt*("
            f"({pivot})+log(max(1e-300,{'+'.join(sum_terms)}))))"
        )
        self.force = openmm.CustomCVForce(energy_expression)
        self.force.addGlobalParameter("kt", kt)
        self.force.addGlobalParameter("beta", beta)
        self.force.addGlobalParameter(f"{self.prefix}_bias_scale", 1.0)
        for index in range(self.n_states):
            self.force.addGlobalParameter(f"{self.prefix}_f_{index}", 0.0)
        self.force.setForceGroup(1)
        self._cv_keeper = []
        self._int_cv_force_xmls = []
        self._basis_cv_indices = []
        for basis_index, basis_force in enumerate(basis_forces):
            self._cv_keeper.append(basis_force)
            cv_index = self.force.addCollectiveVariable(
                f"neural_basis_{basis_index}", basis_force
            )
            self._basis_cv_indices.append(cv_index)
        self.neural_path_protocol_sha256 = controller.protocol_sha256(
            lambdas=self.lambdas
        )

    def addCollectiveVariable(self, name: str, cv_force: Any) -> int:
        variable_name = _nonempty_string(name, "collective variable name")
        if variable_name.startswith("neural_basis_"):
            raise NeuralPathConfigError(
                "neural_basis_* 名称由 OuterLambdaIBSBiasForce 保留"
            )
        self._cv_keeper.append(cv_force)
        if variable_name.endswith("_int"):
            openmm, _ = _require_openmm()
            self._int_cv_force_xmls.append(
                openmm.XmlSerializer.serialize(cv_force)
            )
        return self.force.addCollectiveVariable(variable_name, cv_force)

    def get_force(self):
        return self.force

    def setForceGroup(self, group_id: int) -> None:
        self.force.setForceGroup(group_id)

    def set_bias_enabled(self, context: Any, enabled: bool) -> None:
        context.setParameter(
            f"{self.prefix}_bias_scale", 1.0 if enabled else 0.0
        )

    def update_parameters(
        self, context: Any, f_values: Sequence[float]
    ) -> None:
        if len(f_values) != self.n_states:
            raise NeuralPathConfigError("f_values 数量与 IBS 状态数不一致")
        for index, value in enumerate(f_values):
            context.setParameter(
                f"{self.prefix}_f_{index}",
                _finite_float(value, f"f_values[{index}]"),
            )

    def get_centered_basis_energies_kj_mol(
        self, context: Any
    ) -> tuple[float, ...]:
        values = self.force.getCollectiveVariableValues(context)
        return tuple(
            _finite_float(values[index], f"neural_basis[{basis_index}]")
            for basis_index, index in enumerate(self._basis_cv_indices)
        )


class OuterLambdaResidualBiasForce:
    """DEC-052 / EXP-013 design (3): the exact residual `dV = V_* - V_0`.

    `V_0 = -kT*log(sum_k exp(-(U_k^0 - f_k)/kT))` is the classical-only IBS
    discriminant (no student); `V_* = -kT*log(sum_k exp(-(U_k^0 + A_k*basis -
    f_k)/kT))` is the DEC-048-validated fused discriminant
    (`OuterLambdaIBSBiasForce`). `dV := V_* - V_0` by definition, so
    `V_0 + dV === V_*` is a *construction* identity, not something that needs
    re-deriving physics for -- putting the UNMODIFIED classical `IBSBiasForce`
    (`V_0`) on a fast MTS force group and this Force (`dV`) on a slow one
    reproduces the exact `N=1` Hamiltonian `OuterLambdaIBSBiasForce` already
    validated (DEC-047/048), letting MTS change only *how often* `dV` gets
    re-evaluated, not *what* Hamiltonian is being sampled.

    The one real cost this design cannot avoid: `CustomCVForce` collective
    variables each get their own inner Context, so this Force cannot reuse
    the classical `cv_k_int`/`cv_k_rest` CV *values* the fast `V_0` group
    already computed -- it must register its OWN fresh copies (same pattern
    `_build_probe_context`/the wiring smoke already use: deserialize from
    `IBSBiasForce._int_cv_force_xmls`, never share a live Force object across
    two different `CustomCVForce` parents) and recompute them internally.
    That redundant classical-CV cost, plus the student TorchForce call, plus
    doing the log-sum-exp math twice, is `dV`'s real per-call cost -- EXP-013
    013-A measures it directly rather than assuming it equals the student's
    cost alone.
    """

    def __init__(
        self,
        controller: OuterLambdaController,
        lambdas: Iterable[float],
        temperature_kelvin: Any,
        basis_forces: Sequence[Any],
        *,
        prefix: str,
    ) -> None:
        if not isinstance(controller, OuterLambdaController):
            raise TypeError("controller 必须是 OuterLambdaController")
        self.controller = controller
        self.lambdas = tuple(_finite_float(value, "lambda") for value in lambdas)
        if not self.lambdas:
            raise NeuralPathConfigError("IBS lambda schedule 不能为空")
        self.n_states = len(self.lambdas)
        self.prefix = _nonempty_string(prefix, "prefix")
        self.controller.validate_cv_budget(self.n_states)
        if len(basis_forces) != self.controller.basis_count:
            raise NeuralPathConfigError("basis_forces 数量必须等于 controller.basis_count")
        openmm, unit = _require_openmm()
        try:
            temperature = temperature_kelvin.value_in_unit(unit.kelvin)
        except AttributeError:
            temperature = _finite_float(temperature_kelvin, "temperature_kelvin")
        if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
            raise NeuralPathConfigError("temperature_kelvin 必须为有限正数")
        kt = (unit.MOLAR_GAS_CONSTANT_R * (float(temperature) * unit.kelvin)).value_in_unit(
            unit.kilojoules_per_mole
        )
        beta = 1.0 / kt
        matrix = self.controller.coefficient_matrix(self.lambdas)

        def state_expression(index: int, include_path: bool) -> str:
            expression = f"cv_{index}_int+cv_{index}_rest"
            if include_path:
                path_terms = [
                    f"({coefficient:.17g})*neural_basis_{basis_index}"
                    for basis_index, coefficient in enumerate(matrix[index])
                    if coefficient != 0.0
                ]
                if path_terms:
                    expression += "+" + "+".join(path_terms)
            expression += f"-{self.prefix}_f_{index}"
            return f"({expression})"

        def log_sum_exp_expression(state_exprs: list[str]) -> str:
            logit_expressions = {
                index: f"(-beta*(({state_exprs[index]})-({state_exprs[0]})))"
                for index in range(1, self.n_states)
            }
            pivot = "0.0"
            for index in range(1, self.n_states):
                pivot = f"max({pivot},{logit_expressions[index]})"
            sum_terms = [f"exp(-({pivot}))"] + [
                f"exp({logit_expressions[index]}-({pivot}))" for index in range(1, self.n_states)
            ]
            return f"({state_exprs[0]}-kt*(({pivot})+log(max(1e-300,{'+'.join(sum_terms)}))))"

        state_expressions_fused = [state_expression(index, True) for index in range(self.n_states)]
        state_expressions_classical = [state_expression(index, False) for index in range(self.n_states)]
        v_fused_expression = log_sum_exp_expression(state_expressions_fused)
        v_classical_expression = log_sum_exp_expression(state_expressions_classical)
        # bias_scale applied ONCE on the outside of the difference -- correct only as long as
        # whatever sets {prefix}_bias_scale/{prefix}_f_{k} keeps this Force and the classical
        # V_0 Force (same prefix, shared Context-level global parameters) in sync; both default
        # to the SAME OpenMM global parameter namespace here by construction (see class docstring).
        energy_expression = (
            f"{self.prefix}_bias_scale*(({v_fused_expression})-({v_classical_expression}))"
        )
        self.force = openmm.CustomCVForce(energy_expression)
        self.force.addGlobalParameter("kt", kt)
        self.force.addGlobalParameter("beta", beta)
        self.force.addGlobalParameter(f"{self.prefix}_bias_scale", 1.0)
        for index in range(self.n_states):
            self.force.addGlobalParameter(f"{self.prefix}_f_{index}", 0.0)
        self._cv_keeper: list[Any] = []
        self._int_cv_force_xmls: list[str] = []
        self._basis_cv_indices: list[int] = []
        for basis_index, basis_force in enumerate(basis_forces):
            self._cv_keeper.append(basis_force)
            cv_index = self.force.addCollectiveVariable(f"neural_basis_{basis_index}", basis_force)
            self._basis_cv_indices.append(cv_index)
        self.neural_path_protocol_sha256 = controller.protocol_sha256(lambdas=self.lambdas)

    def addCollectiveVariable(self, name: str, cv_force: Any) -> int:
        # Mirrors OuterLambdaIBSBiasForce.addCollectiveVariable's `_int`-suffix XML
        # stashing (not needed by 013-A itself, but IBSSampler._build_probe_context
        # duck-types on this exact attribute -- keeping it here means this class is
        # already ready for 013-B/C's IBSSampler-based energy collection later).
        variable_name = _nonempty_string(name, "collective variable name")
        if variable_name.startswith("neural_basis_"):
            raise NeuralPathConfigError("neural_basis_* 名称由构造函数保留")
        self._cv_keeper.append(cv_force)
        if variable_name.endswith("_int"):
            openmm, _ = _require_openmm()
            self._int_cv_force_xmls.append(openmm.XmlSerializer.serialize(cv_force))
        return self.force.addCollectiveVariable(variable_name, cv_force)

    def get_force(self):
        return self.force

    def setForceGroup(self, group_id: int) -> None:
        self.force.setForceGroup(group_id)

    def set_bias_enabled(self, context: Any, enabled: bool) -> None:
        context.setParameter(f"{self.prefix}_bias_scale", 1.0 if enabled else 0.0)

    def update_parameters(self, context: Any, f_values: Sequence[float]) -> None:
        if len(f_values) != self.n_states:
            raise NeuralPathConfigError("f_values 数量与 IBS 状态数不一致")
        for index, value in enumerate(f_values):
            context.setParameter(f"{self.prefix}_f_{index}", _finite_float(value, f"f_values[{index}]"))

    def get_centered_basis_energies_kj_mol(self, context: Any) -> tuple[float, ...]:
        values = self.force.getCollectiveVariableValues(context)
        return tuple(
            _finite_float(values[index], f"neural_basis[{basis_index}]")
            for basis_index, index in enumerate(self._basis_cv_indices)
        )

    def validate_wiring(self) -> None:
        """Fail closed BEFORE any Context is created if this Force's
        CustomCVForce is missing a collective variable or global parameter
        the constructed expression actually references, or has a duplicate
        name registered.

        Why this exists (EXP-025 G4 Layer-1, 2026-08-13): this class's own
        constructor registers ONLY the shared basis CV(s) -- by design, the
        CALLER must separately register cv_{k}_int/cv_{k}_rest per state
        (see this class's docstring). The first Layer-1 oracle script omitted
        that registration loop entirely; the resulting CustomCVForce
        referenced undefined cv_k_int/cv_k_rest symbols, yet Context
        construction and every subsequent energy query completed WITHOUT
        raising -- they silently returned a wrong, finite energy (~2.27
        kJ/mol off on the real production window). This method does not
        parse the expression string (Lepton's own validation apparently does
        not reliably catch this class of omission either, or does so lazily
        in a way that never actually fired here) -- it instead compares the
        SET of names this constructor's own contract requires against what
        is ACTUALLY registered on the CustomCVForce object, which is exact
        and needs no parser.
        """
        expected_cv_names = {f"cv_{index}_int" for index in range(self.n_states)}
        expected_cv_names |= {f"cv_{index}_rest" for index in range(self.n_states)}
        expected_cv_names |= {f"neural_basis_{basis_index}" for basis_index in range(len(self._basis_cv_indices))}

        actual_cv_names: list[str] = [
            self.force.getCollectiveVariableName(i) for i in range(self.force.getNumCollectiveVariables())
        ]
        if len(actual_cv_names) != len(set(actual_cv_names)):
            duplicates = sorted({name for name in actual_cv_names if actual_cv_names.count(name) > 1})
            raise NeuralPathConfigError(
                f"OuterLambdaResidualBiasForce.validate_wiring: duplicate collective variable name(s) {duplicates}"
            )
        actual_cv_set = set(actual_cv_names)
        if actual_cv_set != expected_cv_names:
            missing = sorted(expected_cv_names - actual_cv_set)
            unexpected = sorted(actual_cv_set - expected_cv_names)
            raise NeuralPathConfigError(
                "OuterLambdaResidualBiasForce.validate_wiring: collective variable set does not match the "
                f"constructor's contract -- missing={missing}, unexpected={unexpected}. Every cv_{{k}}_int/"
                "cv_{k}_rest for k in range(n_states) must be registered by the caller via addCollectiveVariable() "
                "before creating any Context (this constructor deliberately does not register them itself)."
            )

        expected_global_names = {"kt", "beta", f"{self.prefix}_bias_scale"}
        expected_global_names |= {f"{self.prefix}_f_{index}" for index in range(self.n_states)}
        actual_global_names = {
            self.force.getGlobalParameterName(i) for i in range(self.force.getNumGlobalParameters())
        }
        missing_globals = expected_global_names - actual_global_names
        if missing_globals:
            raise NeuralPathConfigError(
                f"OuterLambdaResidualBiasForce.validate_wiring: missing required global parameter(s) "
                f"{sorted(missing_globals)}"
            )


def evaluate_outer_lambda_force_group_states(
    context,
    lambdas: Iterable[float],
    *,
    lambda_parameter_name: str = "outer_lambda",
    force_group: int = 31,
) -> tuple[float, ...]:
    """在同一 Context 查询所有 λ 的路径 Force-group 能量并恢复原参数。"""

    parameter_name = _nonempty_string(
        lambda_parameter_name, "lambda_parameter_name"
    )
    if (
        isinstance(force_group, bool)
        or not isinstance(force_group, int)
        or force_group < 0
        or force_group > 31
    ):
        raise NeuralPathConfigError("force_group 必须是 [0,31] 整数")
    lambda_values = tuple(
        _finite_float(value, "lambda") for value in lambdas
    )
    if not lambda_values:
        raise NeuralPathConfigError("lambda schedule 不能为空")
    if any(value < 0.0 or value > 1.0 for value in lambda_values):
        raise NeuralPathConfigError("所有 lambda 必须位于 [0,1]")
    try:
        original_value = float(context.getParameter(parameter_name))
    except Exception as exc:
        raise TorchForceDeploymentError(
            f"Context 不含全局参数 {parameter_name!r}: {exc}"
        ) from exc
    energies = []
    try:
        for lam in lambda_values:
            context.setParameter(parameter_name, lam)
            state = context.getState(
                getEnergy=True,
                groups=1 << force_group,
            )
            energy = state.getPotentialEnergy()
            try:
                from openmm import unit

                energy = energy.value_in_unit(unit.kilojoules_per_mole)
            except AttributeError:
                pass
            energies.append(
                _finite_float(energy, f"path_state_energy[{len(energies)}]")
            )
    except Exception as exc:
        if isinstance(
            exc,
            (
                NeuralPathConfigError,
                NeuralPathIntegrityError,
                NeuralPathFrameError,
                TorchForceDeploymentError,
            ),
        ):
            raise
        raise TorchForceDeploymentError(
            f"cross-state path energy 查询失败: {exc}"
        ) from exc
    finally:
        context.setParameter(parameter_name, original_value)
    return tuple(energies)



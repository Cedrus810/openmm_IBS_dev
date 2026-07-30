#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""外层 λ 神经基势的独立、无 ML 运行时依赖核心模块。

本模块实现文档中 WP-1 的数学与协议契约，并提供 WP-2 可复用的能量账本组合函数：

    H~_lambda(R) = H0_lambda(R)
                 + w(lambda) * sum_m c_m(lambda) * (U_m(R) - b_m)

当前协议版本只接受：

* ``sin2`` 包络：``sin(pi*lambda)**2``；
* 常数系数；
* 一个冻结基势（M=1）；
* λ 位于闭区间 [0, 1]。

这里刻意不导入 OpenMM、Torch、MACE 或 NumPy。模型推理和 Force 构建属于后续部署层；
本文件只负责 fail-closed 配置验证、逐文件 SHA-256、确定性系数、稳定协议指纹，以及
明确区分 target energy 与 sampling bias 的账本组合。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


NEURAL_PATH_PROTOCOL_VERSION = 1
NEURAL_BASIS_MODEL_PROTOCOL_VERSION = 1
NEURAL_PATH_ACCOUNTING_VERSION = 1

_SHA256_HEX_LENGTH = 64
_MAX_CUSTOM_CV_VARIABLES = 32


class NeuralPathConfigError(ValueError):
    """神经路径配置违反协议时抛出。"""


class NeuralPathIntegrityError(NeuralPathConfigError):
    """模型、原子选择等外部文件与声明哈希不一致时抛出。"""


class NeuralPathFrameError(RuntimeError):
    """一帧 target/bias/base 账本无法原子提交时抛出。"""


class TorchForceDeploymentError(RuntimeError):
    """独立 TorchForce 构建、序列化或 Context 验证失败时抛出。"""


def _finite_float(value: Any, field: str) -> float:
    """转换成有限 float；bool 不作为数值接受。"""

    if isinstance(value, bool):
        raise NeuralPathConfigError(f"{field} 必须是有限数值，不能是 bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NeuralPathConfigError(f"{field} 必须是有限数值，收到 {value!r}") from exc
    if not math.isfinite(result):
        raise NeuralPathConfigError(f"{field} 必须是有限数值，收到 {value!r}")
    return result


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NeuralPathConfigError(f"{field} 必须是非空字符串")
    return value.strip()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise NeuralPathConfigError(f"协议 payload 不能稳定序列化: {exc}") from exc
    return text.encode("utf-8")


def stable_payload_sha256(payload: Mapping[str, Any]) -> str:
    """返回与映射插入顺序无关的规范 JSON SHA-256。"""

    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """流式计算文件 SHA-256，不依据文件名或 mtime 判断模型身份。"""

    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise NeuralPathIntegrityError(f"需要哈希的文件不存在或不是普通文件: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_sha256(value: Any, field: str) -> str:
    digest = _nonempty_string(value, field).lower()
    if len(digest) != _SHA256_HEX_LENGTH:
        raise NeuralPathConfigError(f"{field} 必须是 64 位 SHA-256 十六进制字符串")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise NeuralPathConfigError(f"{field} 不是合法 SHA-256 十六进制字符串") from exc
    return digest


def _absolute_file_path(value: Any, field: str) -> Path:
    raw = _nonempty_string(value, field)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise NeuralPathConfigError(f"{field} 必须是绝对路径，收到 {raw!r}")
    return path


@dataclass(frozen=True)
class NeuralBasisModelSpec:
    """一个冻结神经基势的最小、可追溯规格。

    ``verify_files=True`` 时会重新计算模型哈希；配置中的 sha256 不是信任根。
    原子选择文件没有外部声明哈希字段，因此其实际内容哈希直接写入协议 payload。
    """

    name: str
    backend: str
    model_path: str
    sha256: str
    energy_offset_kj_mol: float
    atom_selection: str
    atom_indices_path: str
    atom_indices_sha256: str
    output_unit: str
    precision: str
    periodic: bool
    model_protocol_version: int = NEURAL_BASIS_MODEL_PROTOCOL_VERSION

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
        *,
        verify_files: bool = True,
    ) -> "NeuralBasisModelSpec":
        if not isinstance(config, Mapping):
            raise NeuralPathConfigError("bases 中的每一项必须是映射")

        name = _nonempty_string(config.get("name"), "basis.name")
        backend = _nonempty_string(config.get("backend"), f"basis[{name}].backend")
        if backend != "torchforce":
            raise NeuralPathConfigError(
                f"basis[{name}].backend={backend!r}；协议 v1 只支持 'torchforce'"
            )

        model_path = _absolute_file_path(
            config.get("model_path"), f"basis[{name}].model_path"
        )
        declared_sha = _normalize_sha256(
            config.get("sha256"), f"basis[{name}].sha256"
        )
        indices_path = _absolute_file_path(
            config.get("atom_indices_path"), f"basis[{name}].atom_indices_path"
        )

        if verify_files:
            actual_sha = sha256_file(model_path)
            if actual_sha != declared_sha:
                raise NeuralPathIntegrityError(
                    f"basis[{name}] 模型 SHA-256 不匹配: "
                    f"声明 {declared_sha}，实际 {actual_sha}"
                )
            indices_sha = sha256_file(indices_path)
        else:
            # 即便调用者延迟文件校验，payload 也不能伪造一个选择文件身份。
            # 使用显式配置哈希；缺失时 fail closed。
            indices_sha = _normalize_sha256(
                config.get("atom_indices_sha256"),
                f"basis[{name}].atom_indices_sha256",
            )

        offset = _finite_float(
            config.get("energy_offset_kj_mol", 0.0),
            f"basis[{name}].energy_offset_kj_mol",
        )
        atom_selection = _nonempty_string(
            config.get("atom_selection"), f"basis[{name}].atom_selection"
        )
        if atom_selection != "fixed_indices":
            raise NeuralPathConfigError(
                f"basis[{name}].atom_selection={atom_selection!r}；"
                "协议 v1 只支持 'fixed_indices'"
            )
        output_unit = _nonempty_string(
            config.get("output_unit"), f"basis[{name}].output_unit"
        )
        if output_unit != "kJ_per_mol":
            raise NeuralPathConfigError(
                f"basis[{name}].output_unit 必须是 'kJ_per_mol'"
            )
        precision = _nonempty_string(
            config.get("precision"), f"basis[{name}].precision"
        )
        if precision not in {"single", "double"}:
            raise NeuralPathConfigError(
                f"basis[{name}].precision 必须是 'single' 或 'double'"
            )
        periodic = config.get("periodic")
        if not isinstance(periodic, bool):
            raise NeuralPathConfigError(f"basis[{name}].periodic 必须是 bool")

        return cls(
            name=name,
            backend=backend,
            model_path=str(model_path),
            sha256=declared_sha,
            energy_offset_kj_mol=offset,
            atom_selection=atom_selection,
            atom_indices_path=str(indices_path),
            atom_indices_sha256=indices_sha,
            output_unit=output_unit,
            precision=precision,
            periodic=periodic,
        )

    def protocol_payload(self) -> dict[str, Any]:
        """返回进入 production/cache 指纹的稳定模型身份。"""

        return {
            "model_protocol_version": self.model_protocol_version,
            "name": self.name,
            "backend": self.backend,
            "model_path": self.model_path,
            "sha256": self.sha256,
            "energy_offset_kj_mol": self.energy_offset_kj_mol,
            "atom_selection": self.atom_selection,
            "atom_indices_path": self.atom_indices_path,
            "atom_indices_sha256": self.atom_indices_sha256,
            "output_unit": self.output_unit,
            "precision": self.precision,
            "periodic": self.periodic,
        }


@dataclass(frozen=True)
class NeuralPathSafety:
    max_abs_basis_energy_kj_mol: float
    max_abs_path_energy_kj_mol: float
    max_force_norm_kj_mol_nm: float
    fail_on_support_domain_violation: bool

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "NeuralPathSafety":
        if not isinstance(config, Mapping):
            raise NeuralPathConfigError("neural_path.safety 必须是映射")
        values = {}
        for field in (
            "max_abs_basis_energy_kj_mol",
            "max_abs_path_energy_kj_mol",
            "max_force_norm_kj_mol_nm",
        ):
            value = _finite_float(config.get(field), f"neural_path.safety.{field}")
            if value <= 0.0:
                raise NeuralPathConfigError(
                    f"neural_path.safety.{field} 必须严格大于 0"
                )
            values[field] = value
        fail_support = config.get("fail_on_support_domain_violation")
        if not isinstance(fail_support, bool):
            raise NeuralPathConfigError(
                "neural_path.safety.fail_on_support_domain_violation 必须是 bool"
            )
        return cls(**values, fail_on_support_domain_violation=fail_support)

    def protocol_payload(self) -> dict[str, Any]:
        return {
            "max_abs_basis_energy_kj_mol": self.max_abs_basis_energy_kj_mol,
            "max_abs_path_energy_kj_mol": self.max_abs_path_energy_kj_mol,
            "max_force_norm_kj_mol_nm": self.max_force_norm_kj_mol_nm,
            "fail_on_support_domain_violation": (
                self.fail_on_support_domain_violation
            ),
        }


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
class IBSEnergyFrame:
    """一帧完整且已通过同步 finite gate 的 IBS 能量账本。

    ``bias_cv_state_energies`` 是实际进入 IBS log-sum-exp/权重学习的各态能量：
    基础 interaction + neural path，不含 Python 侧 LRC。

    ``target_state_energies`` 是落入 production energy history/TMBAR 的各态能量：
    基础 interaction + neural path + LRC。

    ``sampling_bias_energy`` 只保存实际 IBS/WCA 采样偏置；神经路径项不会进入该字段。
    """

    bias_cv_state_energies_kj_mol: tuple[float, ...]
    target_state_energies_kj_mol: tuple[float, ...]
    neural_path_state_energies_kj_mol: tuple[float, ...]
    basis_energies_kj_mol: tuple[float, ...]
    sampling_bias_energy_kj_mol: float
    base_energy_kj_mol: float


def compose_ibs_energy_frame(
    controller: OuterLambdaController,
    *,
    lambdas: Iterable[float],
    original_interaction_energies_kj_mol: Sequence[float],
    lrc_state_energies_kj_mol: Sequence[float],
    basis_energies_kj_mol: Sequence[float],
    sampling_bias_energy_kj_mol: float,
    base_energy_kj_mol: float,
) -> IBSEnergyFrame:
    """按一次同步 hard gate 形成不可分割的一帧 IBS 账本。"""

    lambda_values = tuple(_finite_float(value, "lambda") for value in lambdas)
    original = tuple(
        _finite_float(value, f"original_interaction[{index}]")
        for index, value in enumerate(original_interaction_energies_kj_mol)
    )
    lrc = tuple(
        _finite_float(value, f"lrc_state_energy[{index}]")
        for index, value in enumerate(lrc_state_energies_kj_mol)
    )
    if not lambda_values:
        raise NeuralPathFrameError("IBS frame 的 lambda schedule 不能为空")
    if not (len(lambda_values) == len(original) == len(lrc)):
        raise NeuralPathFrameError(
            "lambda/original interaction/LRC 的状态数量必须一致"
        )

    sampling_bias = _finite_float(
        sampling_bias_energy_kj_mol, "sampling_bias_energy"
    )
    base = _finite_float(base_energy_kj_mol, "base_energy")

    try:
        neural = controller.neural_path_state_energies(
            lambda_values, basis_energies_kj_mol
        )
        bias_cv = tuple(
            interaction + path
            for interaction, path in zip(original, neural, strict=True)
        )
        target = tuple(
            interaction + path + tail
            for interaction, path, tail in zip(
                original, neural, lrc, strict=True
            )
        )
        basis = (
            tuple(
                _finite_float(value, f"basis_energy[{index}]")
                for index, value in enumerate(basis_energies_kj_mol)
            )
            if controller.enabled
            else ()
        )
    except NeuralPathConfigError as exc:
        raise NeuralPathFrameError(f"IBS frame 神经路径组合失败: {exc}") from exc

    all_state_values = bias_cv + target + neural
    if any(not math.isfinite(value) for value in all_state_values):
        raise NeuralPathFrameError("IBS frame 任一能量分量非有限")

    return IBSEnergyFrame(
        bias_cv_state_energies_kj_mol=bias_cv,
        target_state_energies_kj_mol=target,
        neural_path_state_energies_kj_mol=neural,
        basis_energies_kj_mol=basis,
        sampling_bias_energy_kj_mol=sampling_bias,
        base_energy_kj_mol=base,
    )


class IBSEnergyLedger:
    """target/bias/base 的原子追加账本。

    所有分量先由 :func:`compose_ibs_energy_frame` 完整构造，随后一次性 append。
    构造失败时六份 history 都保持原长度，不允许用零替代坏分量。
    """

    def __init__(self, controller: OuterLambdaController):
        if not isinstance(controller, OuterLambdaController):
            raise TypeError("controller 必须是 OuterLambdaController")
        self.controller = controller
        self.bias_cv_energy_history: list[tuple[float, ...]] = []
        self.target_energy_history: list[tuple[float, ...]] = []
        self.neural_path_energy_history: list[tuple[float, ...]] = []
        self.basis_energy_history: list[tuple[float, ...]] = []
        self.sampling_bias_history: list[float] = []
        self.base_energy_history: list[float] = []

    def __len__(self) -> int:
        lengths = self.history_lengths()
        if len(set(lengths.values())) != 1:
            raise NeuralPathFrameError(
                f"IBS ledger histories 已错位: {lengths}"
            )
        return next(iter(lengths.values()))

    def history_lengths(self) -> dict[str, int]:
        return {
            "bias_cv": len(self.bias_cv_energy_history),
            "target": len(self.target_energy_history),
            "neural_path": len(self.neural_path_energy_history),
            "basis": len(self.basis_energy_history),
            "sampling_bias": len(self.sampling_bias_history),
            "base": len(self.base_energy_history),
        }

    def append_frame(self, **components: Any) -> IBSEnergyFrame:
        frame = compose_ibs_energy_frame(self.controller, **components)
        before = self.history_lengths()
        if len(set(before.values())) != 1:
            raise NeuralPathFrameError(
                f"IBS ledger 在追加前已经错位: {before}"
            )

        self.bias_cv_energy_history.append(
            frame.bias_cv_state_energies_kj_mol
        )
        self.target_energy_history.append(frame.target_state_energies_kj_mol)
        self.neural_path_energy_history.append(
            frame.neural_path_state_energies_kj_mol
        )
        self.basis_energy_history.append(frame.basis_energies_kj_mol)
        self.sampling_bias_history.append(frame.sampling_bias_energy_kj_mol)
        self.base_energy_history.append(frame.base_energy_kj_mol)

        after = self.history_lengths()
        if len(set(after.values())) != 1:
            # 正常 Python list append 不会到这里；若自定义 list/并发篡改破坏原子性，
            # 立即回滚到追加前长度，拒绝留下可被 TMBAR 消费的错位数据。
            for history_name, history in (
                ("bias_cv", self.bias_cv_energy_history),
                ("target", self.target_energy_history),
                ("neural_path", self.neural_path_energy_history),
                ("basis", self.basis_energy_history),
                ("sampling_bias", self.sampling_bias_history),
                ("base", self.base_energy_history),
            ):
                del history[before[history_name] :]
            raise NeuralPathFrameError(
                f"IBS ledger 原子追加失败，已同步回滚: {after}"
            )
        return frame


@dataclass(frozen=True)
class OpenMMPathEvaluation:
    """独立 OpenMM Context 对一个外层路径 Force 的能量/力评价结果。"""

    lambda_value: float
    energy_kj_mol: float
    forces_kj_mol_nm: tuple[tuple[float, float, float], ...]
    max_force_norm_kj_mol_nm: float
    platform_name: str


def _require_openmm():
    """惰性导入 OpenMM；普通控制器/账本使用不会触发任何 OpenMM 初始化。"""

    try:
        import openmm
        from openmm import unit
    except ImportError as exc:
        raise TorchForceDeploymentError(
            "独立 OpenMM 验证需要安装 openmm"
        ) from exc
    return openmm, unit


def build_torchforce_from_spec(spec: NeuralBasisModelSpec):
    """从已验证规格构建冻结 TorchForce，不改变任何生产 System。

    构建前重新计算模型 SHA-256，防止规格创建后模型文件被原地覆盖。返回值是一个
    尚未加入任何 ``System`` 的独立 Force，由调用方拥有。
    """

    if not isinstance(spec, NeuralBasisModelSpec):
        raise TypeError("spec 必须是 NeuralBasisModelSpec")
    if spec.backend != "torchforce":
        raise TorchForceDeploymentError(
            f"backend={spec.backend!r} 不能构建 TorchForce"
        )
    actual_sha = sha256_file(spec.model_path)
    if actual_sha != spec.sha256:
        raise NeuralPathIntegrityError(
            f"TorchForce 构建前模型 SHA-256 已变化: "
            f"声明 {spec.sha256}，实际 {actual_sha}"
        )
    try:
        from openmmtorch import TorchForce
    except ImportError as exc:
        raise TorchForceDeploymentError(
            "构建 TorchForce 需要安装 openmm-torch（Python 模块 openmmtorch）"
        ) from exc
    try:
        force = TorchForce(spec.model_path)
        force.setUsesPeriodicBoundaryConditions(bool(spec.periodic))
        # 当前 openmm-torch 默认让 autograd 从能量产生力；显式声明可防未来默认变化。
        if hasattr(force, "setOutputsForces"):
            force.setOutputsForces(False)
    except Exception as exc:
        raise TorchForceDeploymentError(
            f"TorchForce 无法加载冻结模型 {spec.model_path}: {exc}"
        ) from exc
    return force


def build_openmm_outer_lambda_force(
    controller: OuterLambdaController,
    lambda_value: float,
    basis_forces: Sequence[Any],
):
    """把 M 个独立 basis Force 组合成一个外层 λ ``CustomCVForce``。

    此函数完全独立于项目主程序。每个 basis Force 只注册一次；λ 只出现在外层解析
    系数中，模型本身不接收 λ。OpenMM Force 对象加入 CustomCV 后所有权转移给返回
    的 Force，调用者不应再把原 basis Force 加入另一个 System。
    """

    if not isinstance(controller, OuterLambdaController):
        raise TypeError("controller 必须是 OuterLambdaController")
    openmm, _unit = _require_openmm()

    lam = _finite_float(lambda_value, "lambda")
    if not controller.enabled:
        if basis_forces:
            raise NeuralPathConfigError(
                "neural path 禁用时不应构建或传入 basis Force"
            )
        return openmm.CustomExternalForce("0")
    if len(basis_forces) != controller.basis_count:
        raise NeuralPathConfigError(
            f"期望 {controller.basis_count} 个 basis Force，"
            f"收到 {len(basis_forces)} 个"
        )

    coefficients = controller.state_coefficients(lam)
    terms = []
    for basis_index, coefficient in enumerate(coefficients):
        offset = controller.bases[basis_index].energy_offset_kj_mol
        terms.append(
            f"({repr(float(coefficient))})*"
            f"(basis_{basis_index}-({repr(float(offset))}))"
        )
    expression = " + ".join(terms) if terms else "0"
    outer_force = openmm.CustomCVForce(expression)
    for basis_index, basis_force in enumerate(basis_forces):
        try:
            outer_force.addCollectiveVariable(
                f"basis_{basis_index}", basis_force
            )
        except Exception as exc:
            raise TorchForceDeploymentError(
                f"basis Force {basis_index} 无法注册到 CustomCVForce: {exc}"
            ) from exc
    return outer_force


def build_torchforce_outer_lambda_force(
    controller: OuterLambdaController,
    lambda_value: float,
):
    """从控制器冻结模型规格构建一个独立的外层 λ TorchForce 组合。"""

    if not controller.enabled:
        return build_openmm_outer_lambda_force(controller, lambda_value, [])
    basis_forces = [
        build_torchforce_from_spec(spec) for spec in controller.bases
    ]
    return build_openmm_outer_lambda_force(
        controller, lambda_value, basis_forces
    )


def serialize_openmm_force(force: Any) -> str:
    """独立验证 Force 是否可 XML 序列化；失败时给出明确异常。"""

    openmm, _unit = _require_openmm()
    try:
        return openmm.XmlSerializer.serialize(force)
    except Exception as exc:
        raise TorchForceDeploymentError(
            f"OpenMM Force XML 序列化失败: {exc}"
        ) from exc


def deserialize_openmm_force(force_xml: str):
    """从 XML 可靠重建独立 Force。"""

    if not isinstance(force_xml, str) or not force_xml.strip():
        raise TorchForceDeploymentError("force_xml 必须是非空字符串")
    openmm, _unit = _require_openmm()
    try:
        return openmm.XmlSerializer.deserialize(force_xml)
    except Exception as exc:
        raise TorchForceDeploymentError(
            f"OpenMM Force XML 反序列化失败: {exc}"
        ) from exc


def evaluate_openmm_outer_lambda_force(
    force: Any,
    *,
    lambda_value: float,
    positions_nm: Sequence[Sequence[float]],
    particle_masses_dalton: Sequence[float] | None = None,
    box_vectors_nm: Sequence[Sequence[float]] | None = None,
    platform_name: str = "Reference",
) -> OpenMMPathEvaluation:
    """在一次性独立 Context 中评价路径能量和力。

    该函数不读取或改变项目生产 Context。它适合 WP-2/WP-3 的解析 Force 与
    TorchForce 端点、梯度、PBC、CPU/CUDA 一致性测试。
    """

    openmm, unit = _require_openmm()
    lam = _finite_float(lambda_value, "lambda")
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
                _finite_float(value, f"position[{atom_index}][{axis}]")
                for axis, value in enumerate(position)
            )
        )
    if not normalized_positions:
        raise NeuralPathConfigError("positions_nm 不能为空")

    if particle_masses_dalton is None:
        masses = (1.0,) * len(normalized_positions)
    else:
        masses = tuple(
            _finite_float(value, f"particle_mass[{index}]")
            for index, value in enumerate(particle_masses_dalton)
        )
        if len(masses) != len(normalized_positions):
            raise NeuralPathConfigError("粒子质量数量必须与坐标数量一致")
        if any(value <= 0.0 for value in masses):
            raise NeuralPathConfigError("粒子质量必须严格大于 0")

    system = openmm.System()
    for mass in masses:
        system.addParticle(mass * unit.dalton)
    if box_vectors_nm is not None:
        if len(box_vectors_nm) != 3:
            raise NeuralPathConfigError("box_vectors_nm 必须包含三个盒向量")
        vectors = []
        for vector_index, vector in enumerate(box_vectors_nm):
            if len(vector) != 3:
                raise NeuralPathConfigError("每个盒向量必须是三维向量")
            xyz = tuple(
                _finite_float(
                    value, f"box_vector[{vector_index}][{axis}]"
                )
                for axis, value in enumerate(vector)
            )
            vectors.append(openmm.Vec3(*xyz) * unit.nanometer)
        system.setDefaultPeriodicBoxVectors(*vectors)
    try:
        system.addForce(force)
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        platform = openmm.Platform.getPlatformByName(
            _nonempty_string(platform_name, "platform_name")
        )
        context = openmm.Context(system, integrator, platform)
        context.setPositions(
            [openmm.Vec3(*xyz) for xyz in normalized_positions]
            * unit.nanometer
        )
        if box_vectors_nm is not None:
            context.setPeriodicBoxVectors(*vectors)
        state = context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        raw_forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        forces = tuple(
            tuple(float(value) for value in raw_forces[atom_index])
            for atom_index in range(len(normalized_positions))
        )
    except Exception as exc:
        raise TorchForceDeploymentError(
            f"独立 OpenMM Context 评价失败 ({platform_name}): {exc}"
        ) from exc
    finally:
        # 显式按 Context -> Integrator -> System 顺序释放，方便一个进程连续跑多个探针。
        if "context" in locals():
            del context
        if "integrator" in locals():
            del integrator

    energy = _finite_float(energy, "OpenMM path energy")
    norms = tuple(
        math.sqrt(math.fsum(component * component for component in vector))
        for vector in forces
    )
    if any(not math.isfinite(value) for value in norms):
        raise TorchForceDeploymentError("独立 OpenMM path force 出现非有限值")
    return OpenMMPathEvaluation(
        lambda_value=lam,
        energy_kj_mol=energy,
        forces_kj_mol_nm=forces,
        max_force_norm_kj_mol_nm=max(norms, default=0.0),
        platform_name=str(platform_name),
    )


def load_neural_path_config(
    path: str | Path,
    *,
    verify_basis_files: bool = True,
) -> OuterLambdaController:
    """从 JSON 或 YAML 文件加载控制器。

    YAML 是可选依赖；环境没有 PyYAML 时仍可使用标准库 JSON。未知后缀先按 JSON
    解析，失败后再尝试 YAML。
    """

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise NeuralPathConfigError(f"神经路径配置文件不存在: {config_path}")
    text = config_path.read_text(encoding="utf-8")

    payload: Any
    json_error: Exception | None = None
    if config_path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise NeuralPathConfigError(f"JSON 配置解析失败: {exc}") from exc
    else:
        try:
            if config_path.suffix.lower() not in {".yaml", ".yml"}:
                payload = json.loads(text)
            else:
                raise json.JSONDecodeError("YAML requested", text, 0)
        except json.JSONDecodeError as exc:
            json_error = exc
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as import_exc:
                raise NeuralPathConfigError(
                    "该配置不是可解析 JSON，读取 YAML 需要安装 PyYAML"
                ) from import_exc
            try:
                payload = yaml.safe_load(text)
            except yaml.YAMLError as yaml_exc:
                message = f"YAML 配置解析失败: {yaml_exc}"
                if config_path.suffix.lower() not in {".yaml", ".yml"}:
                    message += f"；此前 JSON 解析也失败: {json_error}"
                raise NeuralPathConfigError(message) from yaml_exc

    if not isinstance(payload, Mapping):
        raise NeuralPathConfigError("配置文件顶层必须是映射")
    return OuterLambdaController.from_mapping(
        payload, verify_basis_files=verify_basis_files
    )


__all__ = [
    "NEURAL_PATH_PROTOCOL_VERSION",
    "NEURAL_BASIS_MODEL_PROTOCOL_VERSION",
    "NEURAL_PATH_ACCOUNTING_VERSION",
    "NeuralPathConfigError",
    "NeuralPathIntegrityError",
    "NeuralPathFrameError",
    "TorchForceDeploymentError",
    "NeuralBasisModelSpec",
    "NeuralPathSafety",
    "OuterLambdaController",
    "AnalyticBasisEvaluation",
    "HarmonicDistanceBasis",
    "IBSEnergyFrame",
    "IBSEnergyLedger",
    "compose_ibs_energy_frame",
    "OpenMMPathEvaluation",
    "build_torchforce_from_spec",
    "build_openmm_outer_lambda_force",
    "build_torchforce_outer_lambda_force",
    "serialize_openmm_force",
    "deserialize_openmm_force",
    "evaluate_openmm_outer_lambda_force",
    "load_neural_path_config",
    "sha256_file",
    "stable_payload_sha256",
]

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""外层 λ 神经基势的独立模块。

覆盖 WP-1/2 的数学和账本契约、WP-3 的 TorchForce 部署探针、WP-4 的固定选择与
支持域门，以及 WP-5 的三臂统计比较：

    H~_lambda(R) = H0_lambda(R)
                 + w(lambda) * sum_m c_m(lambda) * (U_m(R) - b_m)

当前协议版本只接受：

* ``sin2`` 包络：``sin(pi*lambda)**2``；
* 常数系数；
* 一个冻结基势（M=1）；
* λ 位于闭区间 [0, 1]。

模块导入时刻意不导入 OpenMM、Torch、MACE 或 NumPy；只有部署函数被调用时才延迟
加载运行时。这样纯协议、账本和统计工具可以单独运行。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
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


def _read_fixed_atom_indices_file(path: str | Path) -> tuple[int, ...]:
    selection_path = Path(path).expanduser()
    if not selection_path.is_file():
        raise NeuralPathIntegrityError(f"原子选择文件不存在: {selection_path}")
    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NeuralPathConfigError(
            f"原子选择 JSON 解析失败 {selection_path}: {exc}"
        ) from exc
    if isinstance(payload, Mapping):
        payload = payload.get("atom_indices")
    if (
        not isinstance(payload, Sequence)
        or isinstance(payload, (str, bytes))
        or not payload
    ):
        raise NeuralPathConfigError(
            "原子选择文件必须是非空整数列表，或含 atom_indices 的对象"
        )
    indices = []
    for position, value in enumerate(payload):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise NeuralPathConfigError(
                f"atom_indices[{position}] 必须是非负整数"
            )
        indices.append(value)
    if len(set(indices)) != len(indices):
        raise NeuralPathConfigError("atom_indices 不允许重复")
    return tuple(indices)


@dataclass(frozen=True)
class NeuralBasisSupportDomain:
    """固定局部区域的几何支持域阈值（全部旋转/平移不变）。"""

    min_pair_distance_nm: float | None = None
    max_pair_distance_nm: float | None = None
    max_radius_of_gyration_nm: float | None = None

    def __post_init__(self) -> None:
        configured = 0
        for field_name in (
            "min_pair_distance_nm",
            "max_pair_distance_nm",
            "max_radius_of_gyration_nm",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            configured += 1
            if not math.isfinite(value) or value <= 0.0:
                raise NeuralPathConfigError(
                    f"support_domain.{field_name} 必须是有限正数"
                )
        if configured == 0:
            raise NeuralPathConfigError(
                "support_domain 至少需要配置一个几何阈值"
            )
        if (
            self.min_pair_distance_nm is not None
            and self.max_pair_distance_nm is not None
            and self.min_pair_distance_nm >= self.max_pair_distance_nm
        ):
            raise NeuralPathConfigError(
                "support_domain 最小 pair distance 必须小于最大值"
            )

    @classmethod
    def from_mapping(
        cls, config: Mapping[str, Any]
    ) -> "NeuralBasisSupportDomain":
        if not isinstance(config, Mapping):
            raise NeuralPathConfigError("basis.support_domain 必须是映射")
        values = {}
        for field_name in (
            "min_pair_distance_nm",
            "max_pair_distance_nm",
            "max_radius_of_gyration_nm",
        ):
            raw = config.get(field_name)
            values[field_name] = (
                None
                if raw is None
                else _finite_float(raw, f"basis.support_domain.{field_name}")
            )
        return cls(**values)

    def protocol_payload(self) -> dict[str, float | None]:
        return {
            "min_pair_distance_nm": self.min_pair_distance_nm,
            "max_pair_distance_nm": self.max_pair_distance_nm,
            "max_radius_of_gyration_nm": self.max_radius_of_gyration_nm,
        }


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
    model_name: str | None = None
    support_domain: NeuralBasisSupportDomain | None = None
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
        if backend not in {"torchforce", "existing_openmmml"}:
            raise NeuralPathConfigError(
                f"basis[{name}].backend={backend!r}；只支持 "
                "'torchforce' 或 'existing_openmmml'"
            )
        model_name = (
            _nonempty_string(
                config.get("model_name"), f"basis[{name}].model_name"
            )
            if backend == "existing_openmmml"
            else None
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
            _read_fixed_atom_indices_file(indices_path)
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
        raw_support_domain = config.get("support_domain")
        support_domain = (
            NeuralBasisSupportDomain.from_mapping(raw_support_domain)
            if raw_support_domain is not None
            else None
        )

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
            model_name=model_name,
            support_domain=support_domain,
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
            "model_name": self.model_name,
            "support_domain": (
                self.support_domain.protocol_payload()
                if self.support_domain is not None
                else None
            ),
        }

    def atom_indices(self) -> tuple[int, ...]:
        """重新读取固定选择；调用时可检测文件在规格创建后被替换/损坏。"""

        actual_sha = sha256_file(self.atom_indices_path)
        if actual_sha != self.atom_indices_sha256:
            raise NeuralPathIntegrityError(
                f"basis[{self.name}] 原子选择 SHA-256 已变化"
            )
        return _read_fixed_atom_indices_file(self.atom_indices_path)


@dataclass(frozen=True)
class NeuralBasisTaskManifest:
    """WP-4 单任务基势的训练身份、元素覆盖与范围声明。"""

    basis_name: str
    target_slow_variable: str
    atom_elements: tuple[str, ...]
    training_data_path: str
    training_data_sha256: str
    includes_exchange_waters: bool
    includes_ions: bool

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
        *,
        verify_training_data: bool = True,
    ) -> "NeuralBasisTaskManifest":
        if not isinstance(config, Mapping):
            raise NeuralPathConfigError("WP-4 task manifest 必须是映射")
        basis_name = _nonempty_string(
            config.get("basis_name"), "task_manifest.basis_name"
        )
        target = _nonempty_string(
            config.get("target_slow_variable"),
            "task_manifest.target_slow_variable",
        )
        raw_elements = config.get("atom_elements")
        if (
            not isinstance(raw_elements, Sequence)
            or isinstance(raw_elements, (str, bytes))
            or not raw_elements
        ):
            raise NeuralPathConfigError(
                "task_manifest.atom_elements 必须是非空元素符号序列"
            )
        elements = []
        for index, raw_element in enumerate(raw_elements):
            element = _nonempty_string(
                raw_element, f"task_manifest.atom_elements[{index}]"
            )
            if (
                len(element) > 3
                or not element.isalpha()
                or not element[0].isupper()
                or (len(element) > 1 and not element[1:].islower())
            ):
                raise NeuralPathConfigError(
                    f"atom_elements[{index}]={element!r} 不是规范元素符号"
                )
            elements.append(element)
        training_path = _absolute_file_path(
            config.get("training_data_path"),
            "task_manifest.training_data_path",
        )
        training_sha = _normalize_sha256(
            config.get("training_data_sha256"),
            "task_manifest.training_data_sha256",
        )
        if verify_training_data:
            actual_sha = sha256_file(training_path)
            if actual_sha != training_sha:
                raise NeuralPathIntegrityError(
                    "WP-4 training data SHA-256 不匹配: "
                    f"声明 {training_sha}，实际 {actual_sha}"
                )
        includes_exchange_waters = config.get("includes_exchange_waters")
        includes_ions = config.get("includes_ions")
        if not isinstance(includes_exchange_waters, bool):
            raise NeuralPathConfigError(
                "task_manifest.includes_exchange_waters 必须是 bool"
            )
        if not isinstance(includes_ions, bool):
            raise NeuralPathConfigError(
                "task_manifest.includes_ions 必须是 bool"
            )
        return cls(
            basis_name=basis_name,
            target_slow_variable=target,
            atom_elements=tuple(elements),
            training_data_path=str(training_path),
            training_data_sha256=training_sha,
            includes_exchange_waters=includes_exchange_waters,
            includes_ions=includes_ions,
        )

    def protocol_payload(self) -> dict[str, Any]:
        return {
            "basis_name": self.basis_name,
            "target_slow_variable": self.target_slow_variable,
            "atom_elements": list(self.atom_elements),
            "training_data_path": self.training_data_path,
            "training_data_sha256": self.training_data_sha256,
            "includes_exchange_waters": self.includes_exchange_waters,
            "includes_ions": self.includes_ions,
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
class SupportDomainEvaluation:
    basis_name: str
    supported: bool
    selected_atom_count: int
    min_pair_distance_nm: float | None
    max_pair_distance_nm: float | None
    radius_of_gyration_nm: float
    violations: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "basis_name": self.basis_name,
            "supported": self.supported,
            "selected_atom_count": self.selected_atom_count,
            "min_pair_distance_nm": self.min_pair_distance_nm,
            "max_pair_distance_nm": self.max_pair_distance_nm,
            "radius_of_gyration_nm": self.radius_of_gyration_nm,
            "violations": list(self.violations),
        }


def _normalize_box_vectors_nm(
    box_vectors_nm: Sequence[Sequence[float]] | None,
) -> tuple[tuple[float, float, float], ...] | None:
    if box_vectors_nm is None:
        return None
    if (
        not isinstance(box_vectors_nm, Sequence)
        or isinstance(box_vectors_nm, (str, bytes))
        or len(box_vectors_nm) != 3
    ):
        raise NeuralPathConfigError("box_vectors_nm 必须包含三个三维向量")
    vectors = []
    for vector_index, vector in enumerate(box_vectors_nm):
        if (
            not isinstance(vector, Sequence)
            or isinstance(vector, (str, bytes))
            or len(vector) != 3
        ):
            raise NeuralPathConfigError(
                f"box_vectors_nm[{vector_index}] 必须是三维向量"
            )
        vectors.append(
            tuple(
                _finite_float(
                    component,
                    f"box_vectors_nm[{vector_index}][{axis}]",
                )
                for axis, component in enumerate(vector)
            )
        )
    return tuple(vectors)


def _inverse_3x3_rows(
    matrix: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    a, b, c = matrix
    determinant = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    if not math.isfinite(determinant) or abs(determinant) < 1.0e-15:
        raise NeuralPathConfigError("周期盒向量矩阵不可逆")
    inverse_det = 1.0 / determinant
    return (
        (
            (b[1] * c[2] - b[2] * c[1]) * inverse_det,
            (a[2] * c[1] - a[1] * c[2]) * inverse_det,
            (a[1] * b[2] - a[2] * b[1]) * inverse_det,
        ),
        (
            (b[2] * c[0] - b[0] * c[2]) * inverse_det,
            (a[0] * c[2] - a[2] * c[0]) * inverse_det,
            (a[2] * b[0] - a[0] * b[2]) * inverse_det,
        ),
        (
            (b[0] * c[1] - b[1] * c[0]) * inverse_det,
            (a[1] * c[0] - a[0] * c[1]) * inverse_det,
            (a[0] * b[1] - a[1] * b[0]) * inverse_det,
        ),
    )


def _minimum_image_displacement(
    displacement: Sequence[float],
    box_rows: Sequence[Sequence[float]],
    inverse_box_rows: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    # OpenMM box vectors are rows: cartesian = fractional @ box_rows.
    fractional = tuple(
        math.fsum(displacement[row] * inverse_box_rows[row][column] for row in range(3))
        for column in range(3)
    )
    wrapped = tuple(value - math.floor(value + 0.5) for value in fractional)
    return tuple(
        math.fsum(wrapped[row] * box_rows[row][column] for row in range(3))
        for column in range(3)
    )


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
            if (
                basis.periodic
                and basis.support_domain is not None
                and box_rows is None
            ):
                raise NeuralPathConfigError(
                    f"basis[{basis.name}] 是 periodic 支持域，必须提供 box_vectors_nm"
                )
            if basis.periodic and box_rows is not None:
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
                    if basis.periodic and box_rows is not None:
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
        key = (
            "outer_lambda_orbv3_decomposition",
            tuple(int(value) for value in ligand_array),
            tuple(int(value) for value in environment_array),
            self.device,
        )
        cache = self._pipeline._orb_ctx_cache
        if key not in cache:
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
        regions = {}
        for label, indices in (
            ("cplx", self.combined_indices),
            ("lig", self.ligand_indices),
            ("env", self.environment_indices),
        ):
            imaged = self._minimum_image_selected(
                positions_nm, list(indices), box_nm
            )
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
        from openmm import unit

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
        path_energy = amplitude * centered_energy
        path_forces = amplitude * basis_forces
        if abs(path_energy) > self.max_abs_path_energy_kj_mol:
            raise NeuralPathFrameError("MACE path energy 超过安全门")
        if (
            path_forces.size
            and float(np.max(np.linalg.norm(path_forces, axis=1)))
            > self.max_force_norm_kj_mol_nm
        ):
            raise NeuralPathFrameError("MACE path force 超过安全门")
        return float(path_energy), path_forces


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


def run_mace_decomposition_nvt_smoke(
    controller: OuterLambdaController,
    base_system,
    *,
    atomic_numbers: Sequence[int],
    ligand_indices: Sequence[int],
    environment_indices: Sequence[int],
    positions_nm: Sequence[Sequence[float]],
    box_vectors_nm: Sequence[Sequence[float]],
    lambda_value: float = 0.5,
    n_steps: int = 10,
    report_interval: int = 1,
    timestep_fs: float = 0.5,
    temperature_kelvin: float = 300.0,
    friction_per_ps: float = 1.0,
    device: str = "cuda",
    platform_name: str = "CUDA",
    random_seed: int = 20260730,
) -> dict[str, Any]:
    """在 base System 深拷贝中运行真实 MACE 路径短 NVT，不修改调用方 System。"""

    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps <= 0:
        raise NeuralPathConfigError("n_steps 必须是正整数")
    if (
        isinstance(report_interval, bool)
        or not isinstance(report_interval, int)
        or report_interval <= 0
    ):
        raise NeuralPathConfigError("report_interval 必须是正整数")
    lam = _finite_float(lambda_value, "lambda_value")
    controller.envelope(lam)
    timestep = _finite_float(timestep_fs, "timestep_fs")
    temperature = _finite_float(temperature_kelvin, "temperature_kelvin")
    friction = _finite_float(friction_per_ps, "friction_per_ps")
    if min(timestep, temperature, friction) <= 0.0:
        raise NeuralPathConfigError(
            "timestep、temperature、friction 必须为正"
        )
    normalized_positions = _normalize_frame_collection([positions_nm])[0]
    box_rows = _normalize_box_vectors_nm(box_vectors_nm)
    if box_rows is None:
        raise NeuralPathConfigError("MACE NVT smoke 必须提供周期盒")
    openmm, unit = _require_openmm()
    try:
        system_xml = openmm.XmlSerializer.serialize(base_system)
        system = openmm.XmlSerializer.deserialize(system_xml)
    except Exception as exc:
        raise TorchForceDeploymentError(
            f"无法复制 base System: {exc}"
        ) from exc
    if system.getNumParticles() != len(normalized_positions):
        raise NeuralPathConfigError(
            "base System 粒子数与 positions_nm 不一致"
        )
    # NVT smoke 明确移除复制体中的 barostat；原 System 不受影响。
    for force_index in reversed(range(system.getNumForces())):
        force_name = type(system.getForce(force_index)).__name__
        if "Barostat" in force_name:
            system.removeForce(force_index)
    used_groups = {
        int(system.getForce(index).getForceGroup())
        for index in range(system.getNumForces())
    }
    available_groups = [
        group for group in reversed(range(32)) if group not in used_groups
    ]
    if not available_groups:
        raise NeuralPathConfigError("base System 已占用全部 32 个 Force group")
    path_force_group = available_groups[0]
    path_force = build_mace_decomposition_python_force(
        controller,
        atomic_numbers=atomic_numbers,
        ligand_indices=ligand_indices,
        environment_indices=environment_indices,
        default_lambda=lam,
        device=device,
        force_group=path_force_group,
    )
    system.addForce(path_force)
    vectors = tuple(
        openmm.Vec3(*vector) * unit.nanometer for vector in box_rows
    )
    system.setDefaultPeriodicBoxVectors(*vectors)
    integrator = openmm.LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        friction / unit.picosecond,
        timestep * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(int(random_seed))
    platform = openmm.Platform.getPlatformByName(platform_name)
    samples = []
    started = time.perf_counter()
    try:
        context = openmm.Context(system, integrator, platform)
        context.setPositions(
            [openmm.Vec3(*position) for position in normalized_positions]
            * unit.nanometer
        )
        context.setPeriodicBoxVectors(*vectors)
        context.setParameter("outer_lambda", lam)
        context.setVelocitiesToTemperature(
            temperature * unit.kelvin, int(random_seed)
        )
        completed = 0
        path_mask = 1 << path_force_group
        base_mask = ((1 << 32) - 1) ^ path_mask
        while completed < n_steps:
            chunk = min(report_interval, n_steps - completed)
            integrator.step(chunk)
            completed += chunk
            base_state = context.getState(
                getEnergy=True, groups=base_mask
            )
            path_state = context.getState(
                getEnergy=True, groups=path_mask
            )
            total_state = context.getState(
                getEnergy=True,
                getForces=True,
                getPositions=True,
            )
            base_energy = base_state.getPotentialEnergy().value_in_unit(
                unit.kilojoules_per_mole
            )
            path_energy = path_state.getPotentialEnergy().value_in_unit(
                unit.kilojoules_per_mole
            )
            total_energy = total_state.getPotentialEnergy().value_in_unit(
                unit.kilojoules_per_mole
            )
            forces = total_state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoules_per_mole / unit.nanometer
            )
            force_norms = [
                math.sqrt(
                    math.fsum(float(component) ** 2 for component in vector)
                )
                for vector in forces
            ]
            current_positions = total_state.getPositions(
                asNumpy=True
            ).value_in_unit(unit.nanometer)
            finite = (
                all(
                    math.isfinite(value)
                    for value in (base_energy, path_energy, total_energy)
                )
                and all(math.isfinite(value) for value in force_norms)
            )
            if not finite:
                raise TorchForceDeploymentError(
                    f"MACE NVT 在 step {completed} 出现非有限值"
                )
            samples.append(
                {
                    "step": completed,
                    "base_energy_kj_mol": float(base_energy),
                    "path_energy_kj_mol": float(path_energy),
                    "total_energy_kj_mol": float(total_energy),
                    "energy_closure_error_kj_mol": float(
                        total_energy - base_energy - path_energy
                    ),
                    "max_total_force_kj_mol_nm": max(
                        force_norms, default=0.0
                    ),
                    "support_domain": [
                        evaluation.payload()
                        for evaluation in controller.evaluate_support_domains(
                            [
                                [
                                    float(component)
                                    for component in position
                                ]
                                for position in current_positions
                            ],
                            box_vectors_nm=box_rows,
                        )
                    ],
                }
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
            f"MACE decomposition NVT smoke 失败: {exc}"
        ) from exc
    finally:
        if "context" in locals():
            del context
        del integrator
    elapsed = time.perf_counter() - started
    return {
        "report_type": "outer_lambda_mace_decomposition_nvt_smoke",
        "report_version": 1,
        "passed": True,
        "platform": platform_name,
        "device": device,
        "lambda": lam,
        "path_force_group": path_force_group,
        "n_steps": n_steps,
        "report_interval": report_interval,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / n_steps,
        "samples": samples,
        "base_energy_kj_mol": summarize_finite_series(
            [sample["base_energy_kj_mol"] for sample in samples]
        ),
        "path_energy_kj_mol": summarize_finite_series(
            [sample["path_energy_kj_mol"] for sample in samples]
        ),
        "total_energy_kj_mol": summarize_finite_series(
            [sample["total_energy_kj_mol"] for sample in samples]
        ),
        "max_total_force_kj_mol_nm": summarize_finite_series(
            [sample["max_total_force_kj_mol_nm"] for sample in samples]
        ),
        "max_energy_closure_error_kj_mol": max(
            (
                abs(sample["energy_closure_error_kj_mol"])
                for sample in samples
            ),
            default=0.0,
        ),
        "protocol_sha256": controller.protocol_sha256(lambdas=[lam]),
    }


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


def _percentile(values: Sequence[float], probability: float) -> float:
    """确定性线性插值分位数，不依赖 NumPy。"""

    if not values:
        raise NeuralPathConfigError("计算分位数时数据不能为空")
    if not 0.0 <= probability <= 1.0:
        raise NeuralPathConfigError("分位数 probability 必须位于 [0,1]")
    ordered = sorted(_finite_float(value, "percentile value") for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_finite_series(values: Sequence[float]) -> dict[str, float | int]:
    """生成实验日志需要的 count/mean/std/P95/max_abs 摘要。"""

    normalized = tuple(
        _finite_float(value, f"series[{index}]")
        for index, value in enumerate(values)
    )
    if not normalized:
        raise NeuralPathConfigError("待汇总序列不能为空")
    return {
        "count": len(normalized),
        "mean": statistics.fmean(normalized),
        "std": statistics.pstdev(normalized),
        "min": min(normalized),
        "max": max(normalized),
        "p05": _percentile(normalized, 0.05),
        "p50": _percentile(normalized, 0.50),
        "p95": _percentile(normalized, 0.95),
        "max_abs": max(abs(value) for value in normalized),
    }


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


def _normalize_frame_collection(
    frames_nm: Sequence[Sequence[Sequence[float]]],
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    if not isinstance(frames_nm, Sequence) or isinstance(
        frames_nm, (str, bytes)
    ):
        raise NeuralPathConfigError("frames_nm 必须是 [N_frames][N_atoms][3]")
    frames = []
    atom_count = None
    for frame_index, frame in enumerate(frames_nm):
        if not isinstance(frame, Sequence) or isinstance(frame, (str, bytes)):
            raise NeuralPathConfigError(f"frame[{frame_index}] 必须是坐标序列")
        normalized_frame = []
        for atom_index, position in enumerate(frame):
            if (
                not isinstance(position, Sequence)
                or isinstance(position, (str, bytes))
                or len(position) != 3
            ):
                raise NeuralPathConfigError(
                    f"frame[{frame_index}][{atom_index}] 必须是三维坐标"
                )
            normalized_frame.append(
                tuple(
                    _finite_float(
                        component,
                        f"frame[{frame_index}][{atom_index}][{axis}]",
                    )
                    for axis, component in enumerate(position)
                )
            )
        if not normalized_frame:
            raise NeuralPathConfigError(f"frame[{frame_index}] 没有原子")
        if atom_count is None:
            atom_count = len(normalized_frame)
        elif len(normalized_frame) != atom_count:
            raise NeuralPathConfigError("所有 frame 的原子数量必须一致")
        frames.append(tuple(normalized_frame))
    if not frames:
        raise NeuralPathConfigError("frames_nm 不能为空")
    return tuple(frames)


def _process_peak_rss_mib() -> float | None:
    """返回当前进程峰值 RSS；不支持 getrusage 的平台返回 None。"""

    try:
        import resource

        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    # Linux/FreeBSD 是 KiB；macOS 是 bytes。
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return raw / divisor


def _reset_torch_cuda_peak_memory_if_available(
    platform_name: str,
) -> bool:
    if str(platform_name).lower() != "cuda":
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        torch.cuda.reset_peak_memory_stats()
        return True
    except (ImportError, RuntimeError):
        return False


def _torch_cuda_memory_report(enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {
            "available": False,
            "measurement_scope": (
                "PyTorch CUDA allocator only; OpenMM/plugin allocations "
                "are not included"
            ),
        }
    try:
        import torch

        return {
            "available": True,
            "device_index": int(torch.cuda.current_device()),
            "peak_allocated_mib": (
                float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            ),
            "peak_reserved_mib": (
                float(torch.cuda.max_memory_reserved()) / (1024.0 * 1024.0)
            ),
            "measurement_scope": (
                "PyTorch CUDA allocator only; OpenMM/plugin allocations "
                "may exist outside this counter"
            ),
        }
    except (ImportError, RuntimeError) as exc:
        return {
            "available": False,
            "error": str(exc),
            "measurement_scope": "PyTorch CUDA allocator query failed",
        }


def benchmark_existing_orb_mace_basis(
    controller: OuterLambdaController,
    *,
    model_name: str,
    device: str,
    lambdas: Iterable[float],
    frames_nm: Sequence[Sequence[Sequence[float]]],
    ligand_indices: Sequence[int],
    environment_indices: Sequence[int],
    atomic_numbers: Sequence[int],
) -> dict[str, Any]:
    """用项目已有 MACE/ORB pipeline 批量标注并组合外层 λ，不改生产 System。"""

    if not controller.enabled or len(controller.bases) != 1:
        raise NeuralPathConfigError("现有 MACE/ORB benchmark 要求 enabled 且 M=1")
    basis_spec = controller.bases[0]
    if basis_spec.backend == "existing_openmmml":
        if basis_spec.model_name != model_name:
            raise NeuralPathConfigError(
                f"CLI model_name={model_name!r} 与配置 "
                f"model_name={basis_spec.model_name!r} 不一致"
            )
    if basis_spec.periodic:
        raise NeuralPathConfigError(
            "现有 Orbv3DEXPFittingPipeline 使用局部真空分解；"
            "该 benchmark 要求 basis.periodic=false"
        )
    frames = _normalize_frame_collection(frames_nm)
    lambda_values = tuple(
        _finite_float(value, "lambda") for value in lambdas
    )
    if not lambda_values:
        raise NeuralPathConfigError("lambda schedule 不能为空")
    for value in lambda_values:
        controller.envelope(value)
    expected_selection = set(basis_spec.atom_indices())
    requested_selection = set(ligand_indices).union(environment_indices)
    if expected_selection != requested_selection:
        raise NeuralPathConfigError(
            "配置 atom_indices 必须恰好等于 ligand_indices ∪ "
            "environment_indices"
        )

    raw_energies = []
    centered_energies = []
    max_basis_forces = []
    frame_reports = []
    started = time.perf_counter()
    with ExistingOrbMaceBasisAdapter(
        model_name=model_name, device=device
    ) as adapter:
        effective_device = adapter.device
        label_mode = adapter.label_mode
        for frame_index, frame in enumerate(frames):
            frame_started = time.perf_counter()
            evaluation = adapter.evaluate(
                frame,
                ligand_indices=ligand_indices,
                environment_indices=environment_indices,
                atomic_numbers=atomic_numbers,
            )
            centered = controller.centered_basis_energies(
                [evaluation.energy_kj_mol]
            )[0]
            path_energies = controller.neural_path_state_energies(
                lambda_values, [evaluation.energy_kj_mol]
            )
            path_force_maxima = []
            for lam in lambda_values:
                path_forces = controller.neural_path_forces(
                    lam, [evaluation.forces_kj_mol_nm]
                )
                path_force_maxima.append(
                    max(
                        (
                            math.sqrt(
                                math.fsum(
                                    component * component
                                    for component in vector
                                )
                            )
                            for vector in path_forces
                        ),
                        default=0.0,
                    )
                )
            support = controller.evaluate_support_domains(frame)
            raw_energies.append(evaluation.energy_kj_mol)
            centered_energies.append(centered)
            max_basis_forces.append(evaluation.max_force_norm_kj_mol_nm)
            frame_reports.append(
                {
                    "frame_index": frame_index,
                    "evaluation_seconds": time.perf_counter() - frame_started,
                    "basis_energy_kj_mol": evaluation.energy_kj_mol,
                    "centered_basis_energy_kj_mol": centered,
                    "max_basis_force_kj_mol_nm": (
                        evaluation.max_force_norm_kj_mol_nm
                    ),
                    "path_energy_kj_mol": list(path_energies),
                    "max_path_force_kj_mol_nm": path_force_maxima,
                    "support_domain": [
                        item.payload() for item in support
                    ],
                }
            )
    elapsed = time.perf_counter() - started
    support_violation_count = sum(
        1
        for frame in frame_reports
        if any(
            not item["supported"] for item in frame["support_domain"]
        )
    )
    return {
        "report_type": "outer_lambda_existing_orb_mace_benchmark",
        "report_version": 1,
        "passed": support_violation_count == 0,
        "model_name": model_name,
        "label_mode": label_mode,
        "requested_device": device,
        "effective_device": effective_device,
        "n_frames": len(frames),
        "n_atoms": len(frames[0]),
        "ligand_indices": list(ligand_indices),
        "environment_indices": list(environment_indices),
        "lambdas": list(lambda_values),
        "elapsed_seconds": elapsed,
        "seconds_per_frame": elapsed / len(frames),
        "basis_energy_kj_mol": summarize_finite_series(raw_energies),
        "centered_basis_energy_kj_mol": summarize_finite_series(
            centered_energies
        ),
        "max_basis_force_kj_mol_nm": summarize_finite_series(
            max_basis_forces
        ),
        "support_domain_violation_count": support_violation_count,
        "basis_model_sha256": [basis_spec.sha256],
        "atom_selection_sha256": [basis_spec.atom_indices_sha256],
        "protocol_sha256": controller.protocol_sha256(
            lambdas=lambda_values
        ),
        "frames": frame_reports,
    }


def benchmark_torchforce_outer_lambda(
    controller: OuterLambdaController,
    *,
    lambdas: Iterable[float],
    frames_nm: Sequence[Sequence[Sequence[float]]],
    particle_masses_dalton: Sequence[float] | None = None,
    box_vectors_nm: Sequence[Sequence[float]] | None = None,
    platform_name: str = "Reference",
) -> dict[str, Any]:
    """在持久 Context 中批量评价每个 λ，生成 WP-3 性能/稳定性报告。

    每个 λ 构建一个独立 Context，但同一 λ 的所有 frame 复用该 Context，避免把
    Context 创建成本误算成逐帧推理成本。该函数不执行 MD。
    """

    if not controller.enabled:
        raise NeuralPathConfigError("benchmark 要求 neural path enabled=true")
    lambda_values = tuple(
        _finite_float(value, "lambda") for value in lambdas
    )
    if not lambda_values:
        raise NeuralPathConfigError("benchmark lambda schedule 不能为空")
    for value in lambda_values:
        controller.envelope(value)
    frames = _normalize_frame_collection(frames_nm)
    atom_count = len(frames[0])
    support_evaluations_by_frame = [
        controller.evaluate_support_domains(
            frame, box_vectors_nm=box_vectors_nm
        )
        for frame in frames
    ]
    support_violation_count = sum(
        1
        for evaluations in support_evaluations_by_frame
        if any(not evaluation.supported for evaluation in evaluations)
    )
    if particle_masses_dalton is None:
        masses = (1.0,) * atom_count
    else:
        masses = tuple(
            _finite_float(value, f"particle_mass[{index}]")
            for index, value in enumerate(particle_masses_dalton)
        )
        if len(masses) != atom_count or any(value <= 0.0 for value in masses):
            raise NeuralPathConfigError(
                "particle_masses_dalton 必须与原子数一致且全部为正"
            )

    openmm, unit = _require_openmm()
    box_vectors = None
    if box_vectors_nm is not None:
        if len(box_vectors_nm) != 3:
            raise NeuralPathConfigError("box_vectors_nm 必须包含三个向量")
        box_vectors = tuple(
            openmm.Vec3(
                *(
                    _finite_float(component, f"box[{i}][{axis}]")
                    for axis, component in enumerate(vector)
                )
            )
            * unit.nanometer
            for i, vector in enumerate(box_vectors_nm)
        )

    state_reports = []
    peak_rss_before_mib = _process_peak_rss_mib()
    cuda_memory_enabled = _reset_torch_cuda_peak_memory_if_available(
        platform_name
    )
    total_context_seconds = 0.0
    total_evaluation_seconds = 0.0
    safety_violation_count = 0
    for lam in lambda_values:
        context_started = time.perf_counter()
        force = build_torchforce_outer_lambda_force(controller, lam)
        system = openmm.System()
        for mass in masses:
            system.addParticle(mass * unit.dalton)
        if box_vectors is not None:
            system.setDefaultPeriodicBoxVectors(*box_vectors)
        system.addForce(force)
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        platform = openmm.Platform.getPlatformByName(platform_name)
        try:
            context = openmm.Context(system, integrator, platform)
            if box_vectors is not None:
                context.setPeriodicBoxVectors(*box_vectors)
            context_seconds = time.perf_counter() - context_started
            total_context_seconds += context_seconds

            energies = []
            max_forces = []
            rms_forces = []
            evaluation_started = time.perf_counter()
            for frame in frames:
                context.setPositions(
                    [openmm.Vec3(*xyz) for xyz in frame] * unit.nanometer
                )
                state = context.getState(getEnergy=True, getForces=True)
                energy = state.getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole
                )
                raw_forces = state.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer
                )
                force_norms = [
                    math.sqrt(
                        math.fsum(float(component) ** 2 for component in vector)
                    )
                    for vector in raw_forces
                ]
                energy = _finite_float(energy, "benchmark energy")
                if any(not math.isfinite(value) for value in force_norms):
                    raise TorchForceDeploymentError(
                        "benchmark 检测到非有限 force"
                    )
                max_force = max(force_norms, default=0.0)
                rms_force = math.sqrt(
                    math.fsum(value * value for value in force_norms)
                    / max(1, len(force_norms))
                )
                energies.append(energy)
                max_forces.append(max_force)
                rms_forces.append(rms_force)
                if controller.safety is not None and (
                    abs(energy)
                    > controller.safety.max_abs_path_energy_kj_mol
                    or max_force
                    > controller.safety.max_force_norm_kj_mol_nm
                ):
                    safety_violation_count += 1
            evaluation_seconds = time.perf_counter() - evaluation_started
            total_evaluation_seconds += evaluation_seconds
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
                f"benchmark λ={lam:g} 失败: {exc}"
            ) from exc
        finally:
            if "context" in locals():
                del context
            del integrator

        state_reports.append(
            {
                "lambda": lam,
                "coefficients": list(controller.state_coefficients(lam)),
                "context_creation_seconds": context_seconds,
                "evaluation_seconds": evaluation_seconds,
                "seconds_per_frame": evaluation_seconds / len(frames),
                "frames_per_second": (
                    len(frames) / evaluation_seconds
                    if evaluation_seconds > 0.0
                    else None
                ),
                "energy_kj_mol": summarize_finite_series(energies),
                "max_force_kj_mol_nm": summarize_finite_series(max_forces),
                "rms_force_kj_mol_nm": summarize_finite_series(rms_forces),
                "raw_energy_kj_mol": energies,
                "raw_max_force_kj_mol_nm": max_forces,
            }
        )

    peak_rss_after_mib = _process_peak_rss_mib()
    return {
        "report_type": "outer_lambda_torchforce_benchmark",
        "report_version": 1,
        "passed": (
            safety_violation_count == 0
            and (
                support_violation_count == 0
                or controller.safety is None
                or not controller.safety.fail_on_support_domain_violation
            )
        ),
        "platform": platform_name,
        "n_frames": len(frames),
        "n_atoms": atom_count,
        "lambdas": list(lambda_values),
        "protocol_sha256": controller.protocol_sha256(
            lambdas=lambda_values
        ),
        "basis_model_sha256": [basis.sha256 for basis in controller.bases],
        "atom_selection_sha256": [
            basis.atom_indices_sha256 for basis in controller.bases
        ],
        "total_context_creation_seconds": total_context_seconds,
        "total_evaluation_seconds": total_evaluation_seconds,
        "overall_seconds_per_state_frame": (
            total_evaluation_seconds / (len(frames) * len(lambda_values))
        ),
        "safety_violation_count": safety_violation_count,
        "support_domain_violation_count": support_violation_count,
        "support_domain": [
            [evaluation.payload() for evaluation in evaluations]
            for evaluations in support_evaluations_by_frame
        ],
        "memory": {
            "process_peak_rss_before_mib": peak_rss_before_mib,
            "process_peak_rss_after_mib": peak_rss_after_mib,
            "process_peak_rss_increment_mib": (
                max(0.0, peak_rss_after_mib - peak_rss_before_mib)
                if peak_rss_before_mib is not None
                and peak_rss_after_mib is not None
                else None
            ),
            "process_measurement_scope": (
                "ru_maxrss high-water mark for the whole process"
            ),
            "cuda": _torch_cuda_memory_report(cuda_memory_enabled),
        },
        "states": state_reports,
    }


def run_torchforce_nvt_smoke(
    controller: OuterLambdaController,
    *,
    lambda_value: float,
    positions_nm: Sequence[Sequence[float]],
    n_steps: int = 1000,
    report_interval: int = 100,
    timestep_fs: float = 1.0,
    temperature_kelvin: float = 300.0,
    friction_per_ps: float = 1.0,
    particle_masses_dalton: Sequence[float] | None = None,
    box_vectors_nm: Sequence[Sequence[float]] | None = None,
    platform_name: str = "Reference",
    random_seed: int = 20260730,
) -> dict[str, Any]:
    """运行只含神经路径 Force 的独立短 NVT 稳定性 smoke test。"""

    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps <= 0:
        raise NeuralPathConfigError("n_steps 必须是正整数")
    if (
        isinstance(report_interval, bool)
        or not isinstance(report_interval, int)
        or report_interval <= 0
    ):
        raise NeuralPathConfigError("report_interval 必须是正整数")
    timestep_fs = _finite_float(timestep_fs, "timestep_fs")
    temperature_kelvin = _finite_float(
        temperature_kelvin, "temperature_kelvin"
    )
    friction_per_ps = _finite_float(friction_per_ps, "friction_per_ps")
    if timestep_fs <= 0 or temperature_kelvin <= 0 or friction_per_ps < 0:
        raise NeuralPathConfigError("NVT 时间步/温度必须为正，摩擦系数必须非负")

    frames = _normalize_frame_collection([positions_nm])
    positions = frames[0]
    atom_count = len(positions)
    masses = (
        (12.0,) * atom_count
        if particle_masses_dalton is None
        else tuple(
            _finite_float(value, f"particle_mass[{index}]")
            for index, value in enumerate(particle_masses_dalton)
        )
    )
    if len(masses) != atom_count or any(value <= 0 for value in masses):
        raise NeuralPathConfigError("NVT 粒子质量必须与坐标一致且为正")

    openmm, unit = _require_openmm()
    force = build_torchforce_outer_lambda_force(controller, lambda_value)
    system = openmm.System()
    for mass in masses:
        system.addParticle(mass * unit.dalton)
    box_vectors = None
    if box_vectors_nm is not None:
        box_vectors = tuple(
            openmm.Vec3(*map(float, vector)) * unit.nanometer
            for vector in box_vectors_nm
        )
        system.setDefaultPeriodicBoxVectors(*box_vectors)
    system.addForce(force)
    integrator = openmm.LangevinMiddleIntegrator(
        temperature_kelvin * unit.kelvin,
        friction_per_ps / unit.picosecond,
        timestep_fs * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(int(random_seed))
    platform = openmm.Platform.getPlatformByName(platform_name)
    samples = []
    support_violation_count = 0
    started = time.perf_counter()
    try:
        context = openmm.Context(system, integrator, platform)
        if box_vectors is not None:
            context.setPeriodicBoxVectors(*box_vectors)
        context.setPositions(
            [openmm.Vec3(*xyz) for xyz in positions] * unit.nanometer
        )
        context.setVelocitiesToTemperature(
            temperature_kelvin * unit.kelvin, int(random_seed)
        )
        completed = 0
        while completed < n_steps:
            chunk = min(report_interval, n_steps - completed)
            integrator.step(chunk)
            completed += chunk
            state = context.getState(
                getEnergy=True, getForces=True, getPositions=True
            )
            potential = state.getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole
            )
            kinetic = state.getKineticEnergy().value_in_unit(
                unit.kilojoule_per_mole
            )
            raw_forces = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
            raw_positions = state.getPositions(asNumpy=True).value_in_unit(
                unit.nanometer
            )
            force_norms = [
                math.sqrt(
                    math.fsum(float(component) ** 2 for component in vector)
                )
                for vector in raw_forces
            ]
            finite = bool(
                math.isfinite(float(potential))
                and math.isfinite(float(kinetic))
                and all(
                    math.isfinite(float(component))
                    for position in raw_positions
                    for component in position
                )
                and all(math.isfinite(value) for value in force_norms)
            )
            support_evaluations = controller.evaluate_support_domains(
                [
                    [float(component) for component in position]
                    for position in raw_positions
                ],
                box_vectors_nm=box_vectors_nm,
            )
            support_supported = all(
                evaluation.supported for evaluation in support_evaluations
            )
            if not support_supported:
                support_violation_count += 1
            samples.append(
                {
                    "step": completed,
                    "potential_energy_kj_mol": float(potential),
                    "kinetic_energy_kj_mol": float(kinetic),
                    "max_force_kj_mol_nm": max(force_norms, default=0.0),
                    "finite": finite,
                    "support_domain_supported": support_supported,
                    "support_domain": [
                        evaluation.payload()
                        for evaluation in support_evaluations
                    ],
                }
            )
            if not finite:
                raise TorchForceDeploymentError(
                    f"NVT smoke 在 step={completed} 出现非有限值"
                )
    except Exception as exc:
        if isinstance(exc, TorchForceDeploymentError):
            raise
        raise TorchForceDeploymentError(f"NVT smoke 失败: {exc}") from exc
    finally:
        if "context" in locals():
            del context
        del integrator
    elapsed = time.perf_counter() - started

    max_force_values = [sample["max_force_kj_mol_nm"] for sample in samples]
    potential_values = [
        sample["potential_energy_kj_mol"] for sample in samples
    ]
    safety_violations = sum(
        1
        for sample in samples
        if controller.safety is not None
        and (
            abs(sample["potential_energy_kj_mol"])
            > controller.safety.max_abs_path_energy_kj_mol
            or sample["max_force_kj_mol_nm"]
            > controller.safety.max_force_norm_kj_mol_nm
        )
    )
    return {
        "report_type": "outer_lambda_torchforce_nvt_smoke",
        "report_version": 1,
        "passed": (
            safety_violations == 0
            and (
                support_violation_count == 0
                or controller.safety is None
                or not controller.safety.fail_on_support_domain_violation
            )
        ),
        "platform": platform_name,
        "lambda": float(lambda_value),
        "n_steps": n_steps,
        "report_interval": report_interval,
        "timestep_fs": timestep_fs,
        "temperature_kelvin": temperature_kelvin,
        "elapsed_seconds": elapsed,
        "steps_per_second": n_steps / elapsed if elapsed > 0 else None,
        "safety_violation_count": safety_violations,
        "support_domain_violation_count": support_violation_count,
        "potential_energy_kj_mol": summarize_finite_series(potential_values),
        "max_force_kj_mol_nm": summarize_finite_series(max_force_values),
        "samples": samples,
        "protocol_sha256": controller.protocol_sha256(
            lambdas=[lambda_value]
        ),
        "basis_model_sha256": [basis.sha256 for basis in controller.bases],
        "atom_selection_sha256": [
            basis.atom_indices_sha256 for basis in controller.bases
        ],
    }


def qualify_wp4_basis(
    controller: OuterLambdaController,
    task_manifest: NeuralBasisTaskManifest,
    benchmark_report: Mapping[str, Any],
    nvt_report: Mapping[str, Any],
    *,
    max_seconds_per_frame: float,
) -> dict[str, Any]:
    """合并 WP-4 静态身份、代表性轨迹和短 NVT，输出 fail-closed 准入报告。"""

    if not controller.enabled or len(controller.bases) != 1:
        raise NeuralPathConfigError("WP-4 v1 qualification 要求启用且严格 M=1")
    if not isinstance(benchmark_report, Mapping):
        raise NeuralPathConfigError("benchmark_report 必须是映射")
    if not isinstance(nvt_report, Mapping):
        raise NeuralPathConfigError("nvt_report 必须是映射")
    budget = _finite_float(max_seconds_per_frame, "max_seconds_per_frame")
    if budget <= 0.0:
        raise NeuralPathConfigError("max_seconds_per_frame 必须为正")

    basis = controller.bases[0]
    selected_atom_count = len(basis.atom_indices())
    seconds_per_frame = _finite_float(
        benchmark_report.get("overall_seconds_per_state_frame"),
        "benchmark.overall_seconds_per_state_frame",
    )
    if seconds_per_frame < 0.0:
        raise NeuralPathConfigError(
            "benchmark.overall_seconds_per_state_frame 必须非负"
        )
    expected_models = [basis.sha256]
    expected_selections = [basis.atom_indices_sha256]
    checks = {
        "basis_name_matches": task_manifest.basis_name == basis.name,
        "fixed_particle_mapping": (
            selected_atom_count == len(task_manifest.atom_elements)
        ),
        "support_domain_configured": basis.support_domain is not None,
        "exchange_waters_excluded": not task_manifest.includes_exchange_waters,
        "ions_excluded": not task_manifest.includes_ions,
        "benchmark_report_type": benchmark_report.get("report_type")
        == "outer_lambda_torchforce_benchmark",
        "benchmark_model_identity": (
            benchmark_report.get("basis_model_sha256") == expected_models
            and benchmark_report.get("atom_selection_sha256")
            == expected_selections
        ),
        "benchmark_passed": benchmark_report.get("passed") is True,
        "benchmark_no_safety_violations": (
            benchmark_report.get("safety_violation_count") == 0
        ),
        "benchmark_no_support_violations": (
            benchmark_report.get("support_domain_violation_count") == 0
        ),
        "inference_cost_within_budget": seconds_per_frame <= budget,
        "nvt_report_type": nvt_report.get("report_type")
        == "outer_lambda_torchforce_nvt_smoke",
        "nvt_model_identity": (
            nvt_report.get("basis_model_sha256") == expected_models
            and nvt_report.get("atom_selection_sha256")
            == expected_selections
        ),
        "nvt_passed": nvt_report.get("passed") is True,
        "nvt_no_safety_violations": (
            nvt_report.get("safety_violation_count") == 0
        ),
        "nvt_no_support_violations": (
            nvt_report.get("support_domain_violation_count") == 0
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    manifest_payload = task_manifest.protocol_payload()
    return {
        "report_type": "outer_lambda_wp4_basis_qualification",
        "report_version": 1,
        "qualified": not failed_checks,
        "checks": checks,
        "failed_checks": failed_checks,
        "basis_name": basis.name,
        "target_slow_variable": task_manifest.target_slow_variable,
        "selected_atom_count": selected_atom_count,
        "element_counts": {
            element: task_manifest.atom_elements.count(element)
            for element in sorted(set(task_manifest.atom_elements))
        },
        "max_seconds_per_frame": budget,
        "observed_seconds_per_frame": seconds_per_frame,
        "task_manifest": manifest_payload,
        "task_manifest_sha256": stable_payload_sha256(manifest_payload),
        "controller_protocol_sha256": controller.protocol_sha256(),
    }


def importance_effective_sample_size(
    log_weights: Sequence[float],
) -> dict[str, float | int]:
    """从未归一化 log importance weights 稳定计算绝对 ESS 与 ESS ratio。"""

    values = tuple(
        _finite_float(value, f"log_weight[{index}]")
        for index, value in enumerate(log_weights)
    )
    if not values:
        raise NeuralPathConfigError("log_weights 不能为空")
    pivot = max(values)
    shifted = tuple(math.exp(value - pivot) for value in values)
    total = math.fsum(shifted)
    square_total = math.fsum(value * value for value in shifted)
    if total <= 0.0 or square_total <= 0.0:
        raise NeuralPathConfigError("importance weights 退化为零")
    ess = total * total / square_total
    return {
        "n_samples": len(values),
        "absolute_ess": ess,
        "ess_ratio": ess / len(values),
        "max_normalized_weight": max(shifted) / total,
    }


def count_discrete_transitions(labels: Sequence[Any]) -> int:
    """统计相邻去抖后离散慢状态的转换次数。"""

    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        raise NeuralPathConfigError("slow_state_labels 必须是序列")
    if len(labels) < 2:
        return 0
    return sum(
        1 for previous, current in zip(labels, labels[1:]) if current != previous
    )


def integrated_autocorrelation_time(
    values: Sequence[float], max_lag: int | None = None
) -> dict[str, float | int]:
    """使用初始正序列截断估计积分自相关时间与统计低效因子。"""

    series = tuple(
        _finite_float(value, f"timeseries[{index}]")
        for index, value in enumerate(values)
    )
    n = len(series)
    if n < 2:
        raise NeuralPathConfigError("自相关估计至少需要两个样本")
    mean = statistics.fmean(series)
    centered = tuple(value - mean for value in series)
    variance = math.fsum(value * value for value in centered) / n
    if variance == 0.0:
        return {
            "n_samples": n,
            "max_lag_used": 0,
            "integrated_autocorrelation_time_frames": 0.5,
            "statistical_inefficiency": 1.0,
            "effective_uncorrelated_samples": float(n),
        }
    if max_lag is None:
        lag_limit = min(n - 1, max(1, int(math.sqrt(n) * 5)))
    else:
        if isinstance(max_lag, bool) or not isinstance(max_lag, int):
            raise NeuralPathConfigError("max_lag 必须是整数")
        lag_limit = min(n - 1, max(1, max_lag))
    positive_sum = 0.0
    used = 0
    for lag in range(1, lag_limit + 1):
        covariance = math.fsum(
            centered[index] * centered[index + lag]
            for index in range(n - lag)
        ) / (n - lag)
        correlation = covariance / variance
        if correlation <= 0.0:
            break
        positive_sum += correlation
        used = lag
    tau = 0.5 + positive_sum
    inefficiency = max(1.0, 2.0 * tau)
    return {
        "n_samples": n,
        "max_lag_used": used,
        "integrated_autocorrelation_time_frames": tau,
        "statistical_inefficiency": inefficiency,
        "effective_uncorrelated_samples": n / inefficiency,
    }


def analyze_wp5_arm(config: Mapping[str, Any]) -> dict[str, Any]:
    """把一组 WP-5 原始指标规范化为可比较报告。"""

    if not isinstance(config, Mapping):
        raise NeuralPathConfigError("WP-5 arm 必须是映射")
    name = _nonempty_string(config.get("name"), "arm.name")
    gpu_hours = _finite_float(config.get("gpu_hours"), f"{name}.gpu_hours")
    if gpu_hours <= 0.0:
        raise NeuralPathConfigError(f"{name}.gpu_hours 必须为正")
    delta_g = _finite_float(
        config.get("delta_g_kj_mol"), f"{name}.delta_g_kj_mol"
    )
    uncertainty = _finite_float(
        config.get("uncertainty_kj_mol"), f"{name}.uncertainty_kj_mol"
    )
    if uncertainty < 0.0:
        raise NeuralPathConfigError(f"{name}.uncertainty_kj_mol 必须非负")

    log_weights = config.get("log_importance_weights")
    if (
        not isinstance(log_weights, Sequence)
        or isinstance(log_weights, (str, bytes))
        or not log_weights
    ):
        raise NeuralPathConfigError(
            f"{name}.log_importance_weights 必须是非空序列"
        )
    ess = importance_effective_sample_size(log_weights)
    labels = config.get("slow_state_labels", [])
    transitions = count_discrete_transitions(labels)
    slow_values = config.get("slow_variable")
    autocorrelation = (
        integrated_autocorrelation_time(slow_values)
        if isinstance(slow_values, Sequence)
        and not isinstance(slow_values, (str, bytes))
        and len(slow_values) >= 2
        else None
    )
    anomaly_count_raw = config.get("anomaly_count", 0)
    if (
        isinstance(anomaly_count_raw, bool)
        or not isinstance(anomaly_count_raw, int)
        or anomaly_count_raw < 0
    ):
        raise NeuralPathConfigError(f"{name}.anomaly_count 必须非负")
    anomaly_count = anomaly_count_raw
    n_frames_raw = config.get("n_frames", ess["n_samples"])
    if (
        isinstance(n_frames_raw, bool)
        or not isinstance(n_frames_raw, int)
        or n_frames_raw <= 0
    ):
        raise NeuralPathConfigError(f"{name}.n_frames 必须是正整数")
    n_frames = n_frames_raw
    if anomaly_count > n_frames:
        raise NeuralPathConfigError(f"{name} anomaly_count/n_frames 无效")

    result = {
        "name": name,
        "delta_g_kj_mol": delta_g,
        "uncertainty_kj_mol": uncertainty,
        "gpu_hours": gpu_hours,
        "absolute_ess": ess["absolute_ess"],
        "ess_ratio": ess["ess_ratio"],
        "ess_per_gpu_hour": ess["absolute_ess"] / gpu_hours,
        "max_normalized_weight": ess["max_normalized_weight"],
        "slow_state_transition_count": transitions,
        "autocorrelation": autocorrelation,
        "anomaly_count": anomaly_count,
        "n_frames": n_frames,
        "anomaly_rate": anomaly_count / n_frames,
        "endpoint_contract_passed": bool(
            config.get("endpoint_contract_passed", False)
        ),
        "accounting_contract_passed": bool(
            config.get("accounting_contract_passed", False)
        ),
        "mechanical_stability_passed": bool(
            config.get("mechanical_stability_passed", False)
        ),
    }
    if "replicate_id" in config:
        result["replicate_id"] = _nonempty_string(
            config.get("replicate_id"), f"{name}.replicate_id"
        )
    return result


def compare_wp5_arms(
    arm_configs: Sequence[Mapping[str, Any]],
    *,
    delta_g_sigma_multiplier: float = 2.0,
    minimum_ess_gpu_improvement_fraction: float = 0.0,
    minimum_unique_gain_over_relayout_fraction: float = 0.05,
    anomaly_rate_tolerance: float = 0.0,
) -> dict[str, Any]:
    """比较 baseline / lambda_relayout / neural_path 三组并执行晋级门。"""

    arms = [analyze_wp5_arm(config) for config in arm_configs]
    by_name = {arm["name"]: arm for arm in arms}
    required = {"baseline", "lambda_relayout", "neural_path"}
    if len(arms) != len(required) or set(by_name) != required:
        raise NeuralPathConfigError(
            f"WP-5 必须且只能包含 {sorted(required)}，收到 {sorted(by_name)}"
        )
    baseline = by_name["baseline"]
    relayout = by_name["lambda_relayout"]
    neural = by_name["neural_path"]
    sigma_multiplier = _finite_float(
        delta_g_sigma_multiplier, "delta_g_sigma_multiplier"
    )
    ess_gain = _finite_float(
        minimum_ess_gpu_improvement_fraction,
        "minimum_ess_gpu_improvement_fraction",
    )
    unique_gain = _finite_float(
        minimum_unique_gain_over_relayout_fraction,
        "minimum_unique_gain_over_relayout_fraction",
    )
    anomaly_tolerance = _finite_float(
        anomaly_rate_tolerance, "anomaly_rate_tolerance"
    )
    if min(sigma_multiplier, ess_gain, unique_gain, anomaly_tolerance) < 0.0:
        raise NeuralPathConfigError("WP-5 比较阈值必须非负")

    combined_sigma = math.sqrt(
        baseline["uncertainty_kj_mol"] ** 2
        + neural["uncertainty_kj_mol"] ** 2
    )
    delta_g_difference = abs(
        neural["delta_g_kj_mol"] - baseline["delta_g_kj_mol"]
    )
    checks = {
        "endpoint_contract": neural["endpoint_contract_passed"],
        "accounting_contract": neural["accounting_contract_passed"],
        "mechanical_stability": neural["mechanical_stability_passed"],
        "delta_g_consistent": delta_g_difference
        <= sigma_multiplier * combined_sigma,
        "ess_per_gpu_hour_improved_vs_baseline": neural["ess_per_gpu_hour"]
        >= baseline["ess_per_gpu_hour"] * (1.0 + ess_gain),
        "gain_not_replaced_by_lambda_relayout": neural["ess_per_gpu_hour"]
        >= relayout["ess_per_gpu_hour"] * (1.0 + unique_gain),
        "slow_transitions_improved": neural["slow_state_transition_count"]
        > max(
            baseline["slow_state_transition_count"],
            relayout["slow_state_transition_count"],
        ),
        "anomaly_rate_not_worse": neural["anomaly_rate"]
        <= baseline["anomaly_rate"] + anomaly_tolerance,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "report_type": "outer_lambda_wp5_three_arm_comparison",
        "report_version": 1,
        "promotion_passed": not failed_checks,
        "checks": checks,
        "failed_checks": failed_checks,
        "delta_g_difference_kj_mol": delta_g_difference,
        "combined_uncertainty_kj_mol": combined_sigma,
        "thresholds": {
            "delta_g_sigma_multiplier": sigma_multiplier,
            "minimum_ess_gpu_improvement_fraction": ess_gain,
            "minimum_unique_gain_over_relayout_fraction": unique_gain,
            "anomaly_rate_tolerance": anomaly_tolerance,
        },
        "arms": arms,
    }


def compare_wp5_replicates(
    arm_configs: Sequence[Mapping[str, Any]],
    *,
    minimum_replicates: int = 3,
    delta_g_sigma_multiplier: float = 2.0,
    minimum_ess_gpu_improvement_fraction: float = 0.0,
    minimum_unique_gain_over_relayout_fraction: float = 0.05,
    anomaly_rate_tolerance: float = 0.0,
) -> dict[str, Any]:
    """汇总配对独立重复并执行 WP-5/production 三臂晋级门。"""

    if (
        isinstance(minimum_replicates, bool)
        or not isinstance(minimum_replicates, int)
        or minimum_replicates <= 0
    ):
        raise NeuralPathConfigError("minimum_replicates 必须是正整数")
    analyzed = [analyze_wp5_arm(config) for config in arm_configs]
    required = ("baseline", "lambda_relayout", "neural_path")
    grouped = {name: [] for name in required}
    for arm in analyzed:
        name = arm["name"]
        if name not in grouped:
            raise NeuralPathConfigError(f"未知 WP-5 arm name: {name!r}")
        if "replicate_id" not in arm:
            raise NeuralPathConfigError(
                f"{name} 的重复缺少 replicate_id"
            )
        grouped[name].append(arm)

    replicate_sets = {}
    for name, repeats in grouped.items():
        identifiers = [repeat["replicate_id"] for repeat in repeats]
        if len(identifiers) < minimum_replicates:
            raise NeuralPathConfigError(
                f"{name} 只有 {len(identifiers)} 个重复，"
                f"至少需要 {minimum_replicates} 个"
            )
        if len(set(identifiers)) != len(identifiers):
            raise NeuralPathConfigError(f"{name} 的 replicate_id 不允许重复")
        replicate_sets[name] = set(identifiers)
    if len({frozenset(values) for values in replicate_sets.values()}) != 1:
        raise NeuralPathConfigError("三臂必须使用完全相同的配对 replicate_id")

    def aggregate(name: str) -> dict[str, Any]:
        repeats = grouped[name]
        n = len(repeats)
        delta_values = [repeat["delta_g_kj_mol"] for repeat in repeats]
        delta_mean = statistics.fmean(delta_values)
        between_sd = statistics.stdev(delta_values) if n > 1 else 0.0
        within_variance_of_mean = (
            math.fsum(
                repeat["uncertainty_kj_mol"] ** 2 for repeat in repeats
            )
            / (n * n)
        )
        total_standard_error = math.sqrt(
            within_variance_of_mean + between_sd * between_sd / n
        )
        total_gpu_hours = math.fsum(
            repeat["gpu_hours"] for repeat in repeats
        )
        total_ess = math.fsum(
            repeat["absolute_ess"] for repeat in repeats
        )
        total_anomalies = sum(repeat["anomaly_count"] for repeat in repeats)
        total_frames = sum(repeat["n_frames"] for repeat in repeats)
        return {
            "name": name,
            "replicate_count": n,
            "replicate_ids": sorted(replicate_sets[name]),
            "delta_g_mean_kj_mol": delta_mean,
            "delta_g_between_replicate_sd_kj_mol": between_sd,
            "delta_g_total_standard_error_kj_mol": total_standard_error,
            "total_absolute_ess": total_ess,
            "total_gpu_hours": total_gpu_hours,
            "ess_per_gpu_hour": total_ess / total_gpu_hours,
            "mean_slow_state_transition_count": statistics.fmean(
                repeat["slow_state_transition_count"] for repeat in repeats
            ),
            "total_anomaly_count": total_anomalies,
            "total_frames": total_frames,
            "anomaly_rate": total_anomalies / total_frames,
            "endpoint_contract_passed": all(
                repeat["endpoint_contract_passed"] for repeat in repeats
            ),
            "accounting_contract_passed": all(
                repeat["accounting_contract_passed"] for repeat in repeats
            ),
            "mechanical_stability_passed": all(
                repeat["mechanical_stability_passed"] for repeat in repeats
            ),
            "replicates": repeats,
        }

    aggregate_by_name = {name: aggregate(name) for name in required}
    baseline = aggregate_by_name["baseline"]
    relayout = aggregate_by_name["lambda_relayout"]
    neural = aggregate_by_name["neural_path"]
    sigma_multiplier = _finite_float(
        delta_g_sigma_multiplier, "delta_g_sigma_multiplier"
    )
    ess_gain = _finite_float(
        minimum_ess_gpu_improvement_fraction,
        "minimum_ess_gpu_improvement_fraction",
    )
    unique_gain = _finite_float(
        minimum_unique_gain_over_relayout_fraction,
        "minimum_unique_gain_over_relayout_fraction",
    )
    anomaly_tolerance = _finite_float(
        anomaly_rate_tolerance, "anomaly_rate_tolerance"
    )
    if min(sigma_multiplier, ess_gain, unique_gain, anomaly_tolerance) < 0.0:
        raise NeuralPathConfigError("WP-5 比较阈值必须非负")
    combined_standard_error = math.sqrt(
        baseline["delta_g_total_standard_error_kj_mol"] ** 2
        + neural["delta_g_total_standard_error_kj_mol"] ** 2
    )
    delta_g_difference = abs(
        neural["delta_g_mean_kj_mol"] - baseline["delta_g_mean_kj_mol"]
    )
    checks = {
        "minimum_paired_replicates": all(
            arm["replicate_count"] >= minimum_replicates
            for arm in aggregate_by_name.values()
        ),
        "endpoint_contract": neural["endpoint_contract_passed"],
        "accounting_contract": neural["accounting_contract_passed"],
        "mechanical_stability": neural["mechanical_stability_passed"],
        "delta_g_consistent": delta_g_difference
        <= sigma_multiplier * combined_standard_error,
        "ess_per_gpu_hour_improved_vs_baseline": neural["ess_per_gpu_hour"]
        >= baseline["ess_per_gpu_hour"] * (1.0 + ess_gain),
        "gain_not_replaced_by_lambda_relayout": neural["ess_per_gpu_hour"]
        >= relayout["ess_per_gpu_hour"] * (1.0 + unique_gain),
        "slow_transitions_improved": neural[
            "mean_slow_state_transition_count"
        ]
        > max(
            baseline["mean_slow_state_transition_count"],
            relayout["mean_slow_state_transition_count"],
        ),
        "anomaly_rate_not_worse": neural["anomaly_rate"]
        <= baseline["anomaly_rate"] + anomaly_tolerance,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "report_type": "outer_lambda_wp5_replicated_comparison",
        "report_version": 1,
        "promotion_passed": not failed_checks,
        "checks": checks,
        "failed_checks": failed_checks,
        "delta_g_difference_kj_mol": delta_g_difference,
        "combined_standard_error_kj_mol": combined_standard_error,
        "minimum_replicates": minimum_replicates,
        "thresholds": {
            "delta_g_sigma_multiplier": sigma_multiplier,
            "minimum_ess_gpu_improvement_fraction": ess_gain,
            "minimum_unique_gain_over_relayout_fraction": unique_gain,
            "anomaly_rate_tolerance": anomaly_tolerance,
        },
        "arms": [aggregate_by_name[name] for name in required],
    }


def resolve_existing_model_artifact(
    model_name: str,
    model_path: str | Path = "auto",
) -> Path:
    """解析现有 MACE/ORB 权重的实际本地文件；ORB auto 会走其官方缓存器。"""

    normalized_name = _nonempty_string(model_name, "model_name")
    if str(model_path) != "auto":
        path = _absolute_file_path(model_path, "model_path")
        if not path.is_file():
            raise NeuralPathIntegrityError(
                f"模型文件不存在或不是普通文件: {path}"
            )
        return path
    if normalized_name == "mace-off24-medium":
        path = Path.home() / ".cache/mace/MACE-OFF24_medium.model"
        if not path.is_file():
            raise NeuralPathIntegrityError(
                "未找到本地 MACE-OFF24 medium；请显式提供 --model-path"
            )
        return path
    if normalized_name.startswith("orb-"):
        if "conservative" not in normalized_name:
            raise NeuralPathConfigError(
                "外层路径只允许自动解析 conservative ORB 模型"
            )
        try:
            import inspect
            from cached_path import cached_path
            from orb_models.forcefield import pretrained

            loader = pretrained.ORB_PRETRAINED_MODELS[normalized_name]
            weights_parameter = inspect.signature(loader).parameters.get(
                "weights_path"
            )
            if (
                weights_parameter is None
                or not isinstance(weights_parameter.default, str)
                or not weights_parameter.default.startswith(("http://", "https://"))
            ):
                raise NeuralPathConfigError(
                    f"无法从 {normalized_name!r} loader 解析官方 weights URL"
                )
            resolved = Path(cached_path(weights_parameter.default)).resolve()
        except KeyError as exc:
            raise NeuralPathConfigError(
                f"orb_models 不认识模型 {normalized_name!r}"
            ) from exc
        except NeuralPathConfigError:
            raise
        except Exception as exc:
            raise NeuralPathIntegrityError(
                f"ORB 权重下载/缓存解析失败 ({normalized_name}): {exc}"
            ) from exc
        if not resolved.is_file():
            raise NeuralPathIntegrityError(
                f"ORB 缓存解析结果不是普通文件: {resolved}"
            )
        return resolved
    raise NeuralPathConfigError(
        f"model_path=auto 尚不支持模型 {normalized_name!r}"
    )


def prepare_existing_model_node_config(
    *,
    selection_meta_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    model_name: str = "mace-off24-medium",
    coefficient: float = 0.1,
    energy_offset_kj_mol: float = 0.0,
    max_abs_basis_energy_kj_mol: float = 5000.0,
    max_abs_path_energy_kj_mol: float = 1000.0,
    max_force_norm_kj_mol_nm: float = 5000.0,
) -> dict[str, Any]:
    """从现有 DEXP meta 生成节点所需的固定选择和独立配置文件。"""

    meta = _cli_read_json_mapping(selection_meta_path, "selection-meta")
    ligand = meta.get("ligand_indices")
    environment = meta.get("env_indices")
    if not isinstance(ligand, list) or not isinstance(environment, list):
        raise NeuralPathConfigError(
            "selection meta 必须包含 ligand_indices 和 env_indices 列表"
        )
    combined = []
    for field, values in (
        ("ligand_indices", ligand),
        ("env_indices", environment),
    ):
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise NeuralPathConfigError(
                    f"{field}[{index}] 必须是非负整数"
                )
            combined.append(value)
    if not ligand or not environment:
        raise NeuralPathConfigError("ligand/environment 原子集合都不能为空")
    if len(set(combined)) != len(combined):
        raise NeuralPathConfigError(
            "ligand/environment 原子集合内部或彼此之间存在重复"
        )

    artifact_path = resolve_existing_model_artifact(model_name, model_path)
    artifact_sha = sha256_file(artifact_path)
    coefficient_value = _finite_float(coefficient, "coefficient")
    offset = _finite_float(energy_offset_kj_mol, "energy_offset_kj_mol")
    basis_energy_limit = _finite_float(
        max_abs_basis_energy_kj_mol,
        "max_abs_basis_energy_kj_mol",
    )
    path_energy_limit = _finite_float(
        max_abs_path_energy_kj_mol,
        "max_abs_path_energy_kj_mol",
    )
    force_limit = _finite_float(
        max_force_norm_kj_mol_nm, "max_force_norm_kj_mol_nm"
    )
    if min(basis_energy_limit, path_energy_limit, force_limit) <= 0.0:
        raise NeuralPathConfigError("三个 safety limit 都必须为正")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    indices_path = destination / "fixed_local_atom_indices.json"
    config_path = destination / "outer_lambda_existing_model.json"
    indices_payload = sorted(combined)
    indices_path.write_text(
        json.dumps(indices_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config_payload = {
        "neural_path": {
            "enabled": True,
            "protocol_version": NEURAL_PATH_PROTOCOL_VERSION,
            "stage": "vanishing",
            "baseline_potential": "softcore",
            "endpoint_tolerance": 1.0e-12,
            "envelope": {"type": "sin2", "parameters": {}},
            "coefficient_model": {
                "type": "constant",
                "coefficients": [coefficient_value],
                "max_abs_coefficient": max(1.0, abs(coefficient_value)),
            },
            "bases": [
                {
                    "name": "existing_local_interaction_basis",
                    "backend": "existing_openmmml",
                    "model_name": _nonempty_string(
                        model_name, "model_name"
                    ),
                    "model_path": str(artifact_path),
                    "sha256": artifact_sha,
                    "energy_offset_kj_mol": offset,
                    "atom_selection": "fixed_indices",
                    "atom_indices_path": str(indices_path),
                    "output_unit": "kJ_per_mol",
                    "precision": "single",
                    "periodic": False,
                }
            ],
            "safety": {
                "max_abs_basis_energy_kj_mol": basis_energy_limit,
                "max_abs_path_energy_kj_mol": path_energy_limit,
                "max_force_norm_kj_mol_nm": force_limit,
                "fail_on_support_domain_violation": True,
            },
        }
    }
    config_path.write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    controller = load_neural_path_config(config_path)
    return {
        "report_type": "outer_lambda_existing_model_node_preparation",
        "report_version": 1,
        "model_name": model_name,
        "model_path": str(artifact_path),
        "model_sha256": artifact_sha,
        "ligand_atom_count": len(ligand),
        "environment_atom_count": len(environment),
        "selected_atom_count": len(indices_payload),
        "selection_meta_path": str(Path(selection_meta_path).resolve()),
        "atom_indices_path": str(indices_path),
        "config_path": str(config_path),
        "protocol_sha256": controller.protocol_sha256(),
    }


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


def _cli_lambda_schedule(value: str) -> tuple[float, ...]:
    """解析 CLI 的逗号分隔 λ schedule。"""

    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError("λ schedule 不能为空")
    try:
        lambdas = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"λ schedule 必须是逗号分隔数值，收到 {value!r}"
        ) from exc
    if not lambdas or any(not math.isfinite(item) for item in lambdas):
        raise argparse.ArgumentTypeError("λ schedule 必须包含有限数值")
    if any(item < 0.0 or item > 1.0 for item in lambdas):
        raise argparse.ArgumentTypeError("所有 λ 必须位于 [0, 1]")
    return lambdas


def _cli_write_json(payload: Mapping[str, Any], output: str | None) -> None:
    text = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if output is None:
        print(text)
        return
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")


def _cli_read_probe_input(path: str | Path) -> dict[str, Any]:
    input_path = Path(path).expanduser()
    if not input_path.is_file():
        raise NeuralPathConfigError(f"probe 输入文件不存在: {input_path}")
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NeuralPathConfigError(f"probe 输入 JSON 解析失败: {exc}") from exc
    if isinstance(payload, list):
        payload = {"positions_nm": payload}
    if not isinstance(payload, dict):
        raise NeuralPathConfigError(
            "probe 输入必须是坐标列表或包含 positions_nm 的 JSON 对象"
        )
    if "positions_nm" not in payload:
        raise NeuralPathConfigError("probe 输入缺少 positions_nm")
    return payload


def _cli_read_json_mapping(path: str | Path, label: str) -> dict[str, Any]:
    input_path = Path(path).expanduser()
    if not input_path.is_file():
        raise NeuralPathConfigError(f"{label} 输入文件不存在: {input_path}")
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NeuralPathConfigError(f"{label} JSON 解析失败: {exc}") from exc
    if not isinstance(payload, dict):
        raise NeuralPathConfigError(f"{label} JSON 顶层必须是对象")
    return payload


def _resolve_trajectory_frame_spec(spec: str, n_frames: int) -> tuple[int, ...]:
    raw = _nonempty_string(spec, "frames")
    if n_frames <= 0:
        raise NeuralPathConfigError("轨迹没有 frame")
    try:
        if raw == "last":
            values = (n_frames - 1,)
        elif raw.startswith("tail:"):
            count = int(raw.split(":", 1)[1])
            if count <= 0:
                raise ValueError
            values = tuple(range(max(0, n_frames - count), n_frames))
        elif ":" in raw:
            fields = raw.split(":")
            if len(fields) not in {2, 3}:
                raise ValueError
            parsed = [int(value) if value else None for value in fields]
            frame_slice = slice(*parsed)
            values = tuple(range(n_frames)[frame_slice])
        else:
            values = tuple(int(value.strip()) for value in raw.split(","))
    except ValueError as exc:
        raise NeuralPathConfigError(
            "frames 必须是 last、tail:N、start:stop:step 或逗号整数列表"
        ) from exc
    if not values:
        raise NeuralPathConfigError("frames 选择结果为空")
    if any(value < 0 or value >= n_frames for value in values):
        raise NeuralPathConfigError(
            f"frames 索引必须位于 [0, {n_frames - 1}]"
        )
    if len(set(values)) != len(values):
        raise NeuralPathConfigError("frames 不允许重复索引")
    return values


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="outer_lambda_neural_basis",
        description=(
            "外层 λ 神经基势独立工具；不读取或修改 ABFE 生产 Context。"
        ),
        epilog=(
            "示例:\n"
            "  python outer_lambda_neural_basis.py validate -c neural_path.yaml\n"
            "  python outer_lambda_neural_basis.py coefficients -c neural_path.yaml "
            "--lambdas 0,0.25,0.5,0.75,1\n"
            "  python outer_lambda_neural_basis.py probe -c neural_path.yaml "
            "--lambda 0.5 --input positions.json\n"
            "  python outer_lambda_neural_basis.py compare -c neural_path.yaml "
            "--input three_arms.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_config(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "-c",
            "--config",
            required=True,
            help="神经路径 JSON/YAML 配置文件",
        )
        command_parser.add_argument(
            "-o",
            "--output",
            help="将 JSON 结果写入文件；默认输出到 stdout",
        )

    validate_parser = subparsers.add_parser(
        "validate",
        help="验证配置、模型/选择文件哈希和协议不变量",
    )
    add_common_config(validate_parser)

    coefficients_parser = subparsers.add_parser(
        "coefficients",
        help="生成指定 λ schedule 的 A[k,m] 系数矩阵",
    )
    add_common_config(coefficients_parser)
    coefficients_parser.add_argument(
        "--lambdas",
        required=True,
        type=_cli_lambda_schedule,
        help="逗号分隔 λ，例如 0,0.25,0.5,0.75,1",
    )

    protocol_parser = subparsers.add_parser(
        "protocol",
        help="输出完整稳定协议 payload 和 SHA-256",
    )
    add_common_config(protocol_parser)
    protocol_parser.add_argument(
        "--lambdas",
        type=_cli_lambda_schedule,
        help="可选逗号分隔 λ；提供后 schedule 和 A 矩阵进入协议指纹",
    )

    probe_parser = subparsers.add_parser(
        "probe",
        help="在独立 OpenMM Context 中评价冻结 TorchForce 路径能量和力",
    )
    add_common_config(probe_parser)
    probe_parser.add_argument(
        "--lambda",
        dest="lambda_value",
        required=True,
        type=float,
        help="要评价的单个 λ，范围 [0,1]",
    )
    probe_parser.add_argument(
        "--input",
        required=True,
        help=(
            "JSON：坐标列表，或含 positions_nm、可选 box_vectors_nm/"
            "particle_masses_dalton 的对象"
        ),
    )
    probe_parser.add_argument(
        "--platform",
        default="Reference",
        help="OpenMM platform，默认 Reference",
    )
    probe_parser.add_argument(
        "--xml-roundtrip",
        action="store_true",
        help="评价前先做 Force XML 序列化/反序列化，验证可靠重建",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="对一批 frame 复用 Context，输出能量/力分位数和推理性能",
    )
    add_common_config(benchmark_parser)
    benchmark_parser.add_argument(
        "--lambdas",
        required=True,
        type=_cli_lambda_schedule,
        help="逗号分隔 λ schedule",
    )
    benchmark_parser.add_argument(
        "--input",
        required=True,
        help=(
            "含 frames_nm 的 JSON；也可提供单帧 positions_nm。"
            "可选 box_vectors_nm/particle_masses_dalton"
        ),
    )
    benchmark_parser.add_argument(
        "--platform", default="Reference", help="OpenMM platform"
    )

    nvt_parser = subparsers.add_parser(
        "nvt-smoke",
        help="运行独立短 NVT，检查非有限能量/力和安全阈值",
    )
    add_common_config(nvt_parser)
    nvt_parser.add_argument(
        "--lambda",
        dest="lambda_value",
        required=True,
        type=float,
    )
    nvt_parser.add_argument("--input", required=True)
    nvt_parser.add_argument("--steps", type=int, default=1000)
    nvt_parser.add_argument("--report-interval", type=int, default=100)
    nvt_parser.add_argument("--timestep-fs", type=float, default=1.0)
    nvt_parser.add_argument("--temperature-k", type=float, default=300.0)
    nvt_parser.add_argument("--friction-per-ps", type=float, default=1.0)
    nvt_parser.add_argument("--seed", type=int, default=20260730)
    nvt_parser.add_argument(
        "--platform", default="Reference", help="OpenMM platform"
    )

    existing_parser = subparsers.add_parser(
        "label-existing",
        help="复用项目已有 MACE/ORB pipeline 批量标注并组合外层 λ",
    )
    add_common_config(existing_parser)
    existing_parser.add_argument("--input", required=True)
    existing_parser.add_argument(
        "--model-name", default="mace-off24-medium"
    )
    existing_parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda"
    )
    existing_parser.add_argument(
        "--lambdas",
        required=True,
        type=_cli_lambda_schedule,
    )

    trajectory_parser = subparsers.add_parser(
        "label-trajectory",
        help="直接读取现有轨迹和 DEXP selection meta，批量运行 MACE/ORB",
    )
    add_common_config(trajectory_parser)
    trajectory_parser.add_argument("--trajectory", required=True)
    trajectory_parser.add_argument("--topology", required=True)
    trajectory_parser.add_argument("--selection-meta", required=True)
    trajectory_parser.add_argument(
        "--frames",
        default="last",
        help="last、tail:N、start:stop:step 或逗号分隔 frame 索引",
    )
    trajectory_parser.add_argument(
        "--model-name", default="mace-off24-medium"
    )
    trajectory_parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda"
    )
    trajectory_parser.add_argument(
        "--lambdas",
        required=True,
        type=_cli_lambda_schedule,
    )

    mace_nvt_parser = subparsers.add_parser(
        "mace-nvt-smoke",
        help="在 base System 复制体中运行每步 MACE 分解 Force 的短 NVT",
    )
    add_common_config(mace_nvt_parser)
    mace_nvt_parser.add_argument("--system-xml", required=True)
    mace_nvt_parser.add_argument("--trajectory", required=True)
    mace_nvt_parser.add_argument("--topology", required=True)
    mace_nvt_parser.add_argument("--selection-meta", required=True)
    mace_nvt_parser.add_argument("--frame", default="last")
    mace_nvt_parser.add_argument(
        "--lambda", dest="lambda_value", type=float, default=0.5
    )
    mace_nvt_parser.add_argument("--steps", type=int, default=10)
    mace_nvt_parser.add_argument(
        "--report-interval", type=int, default=1
    )
    mace_nvt_parser.add_argument("--timestep-fs", type=float, default=0.5)
    mace_nvt_parser.add_argument(
        "--temperature-k", type=float, default=300.0
    )
    mace_nvt_parser.add_argument(
        "--friction-per-ps", type=float, default=1.0
    )
    mace_nvt_parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda"
    )
    mace_nvt_parser.add_argument("--platform", default="CUDA")
    mace_nvt_parser.add_argument("--seed", type=int, default=20260730)

    prepare_parser = subparsers.add_parser(
        "prepare-existing",
        help="从现有 DEXP selection meta 生成节点配置和固定原子选择",
    )
    prepare_parser.add_argument("--selection-meta", required=True)
    prepare_parser.add_argument(
        "--model-path",
        default="auto",
        help="权重文件绝对路径；auto 自动解析 MACE/ORB 官方缓存",
    )
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument(
        "--model-name", default="mace-off24-medium"
    )
    prepare_parser.add_argument("--coefficient", type=float, default=0.1)
    prepare_parser.add_argument(
        "--energy-offset-kj-mol", type=float, default=0.0
    )
    prepare_parser.add_argument(
        "--max-abs-basis-energy-kj-mol", type=float, default=5000.0
    )
    prepare_parser.add_argument(
        "--max-abs-path-energy-kj-mol", type=float, default=1000.0
    )
    prepare_parser.add_argument(
        "--max-force-norm-kj-mol-nm", type=float, default=5000.0
    )
    prepare_parser.add_argument(
        "-o", "--output", help="可选 JSON 准备报告"
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="比较 WP-5 baseline / λ 重排 / neural path 三臂指标并执行晋级门",
    )
    add_common_config(compare_parser)
    compare_parser.add_argument(
        "--input",
        required=True,
        help=(
            "JSON 对象：arms 为三臂原始指标列表；可选 thresholds 覆盖晋级阈值"
        ),
    )

    replicated_parser = subparsers.add_parser(
        "compare-replicates",
        help="汇总至少三个配对独立重复并执行 WP-5/production 晋级门",
    )
    add_common_config(replicated_parser)
    replicated_parser.add_argument("--input", required=True)
    replicated_parser.add_argument(
        "--minimum-replicates", type=int, default=3
    )

    qualify_parser = subparsers.add_parser(
        "qualify",
        help="合并任务 manifest、批量 benchmark 与短 NVT，执行 WP-4 准入门",
    )
    add_common_config(qualify_parser)
    qualify_parser.add_argument("--manifest", required=True)
    qualify_parser.add_argument("--benchmark-report", required=True)
    qualify_parser.add_argument("--nvt-report", required=True)
    qualify_parser.add_argument(
        "--max-seconds-per-frame",
        required=True,
        type=float,
        help="允许的单 λ 单 frame 最大推理秒数",
    )
    return parser


def _run_cli_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "prepare-existing":
        report = prepare_existing_model_node_config(
            selection_meta_path=args.selection_meta,
            model_path=args.model_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            coefficient=args.coefficient,
            energy_offset_kj_mol=args.energy_offset_kj_mol,
            max_abs_basis_energy_kj_mol=(
                args.max_abs_basis_energy_kj_mol
            ),
            max_abs_path_energy_kj_mol=args.max_abs_path_energy_kj_mol,
            max_force_norm_kj_mol_nm=args.max_force_norm_kj_mol_nm,
        )
        report["ok"] = True
        report["command"] = "prepare-existing"
        return report

    controller = load_neural_path_config(args.config, verify_basis_files=True)

    if args.command == "validate":
        return {
            "ok": True,
            "command": "validate",
            "config": str(Path(args.config).expanduser().resolve()),
            "enabled": controller.enabled,
            "protocol_version": controller.protocol_version,
            "basis_count": controller.basis_count,
            "basis_names": [basis.name for basis in controller.bases],
            "model_sha256": [basis.sha256 for basis in controller.bases],
            "atom_selection_sha256": [
                basis.atom_indices_sha256 for basis in controller.bases
            ],
            "protocol_sha256": controller.protocol_sha256(),
        }

    if args.command == "coefficients":
        lambdas = tuple(args.lambdas)
        matrix = controller.coefficient_matrix(lambdas)
        return {
            "ok": True,
            "command": "coefficients",
            "enabled": controller.enabled,
            "lambdas": list(lambdas),
            "basis_count": controller.basis_count,
            "coefficient_matrix": [list(row) for row in matrix],
            "protocol_sha256": controller.protocol_sha256(lambdas=lambdas),
        }

    if args.command == "protocol":
        lambdas = tuple(args.lambdas) if args.lambdas is not None else None
        payload = controller.protocol_payload(lambdas=lambdas)
        return {
            "ok": True,
            "command": "protocol",
            "protocol_sha256": stable_payload_sha256(payload),
            "payload": payload,
        }

    if args.command == "probe":
        lam = _finite_float(args.lambda_value, "lambda")
        if lam < 0.0 or lam > 1.0:
            raise NeuralPathConfigError("probe --lambda 必须位于 [0, 1]")
        probe_input = _cli_read_probe_input(args.input)
        force = build_torchforce_outer_lambda_force(controller, lam)
        xml_sha256 = None
        if args.xml_roundtrip:
            force_xml = serialize_openmm_force(force)
            xml_sha256 = hashlib.sha256(force_xml.encode("utf-8")).hexdigest()
            force = deserialize_openmm_force(force_xml)
        result = evaluate_openmm_outer_lambda_force(
            force,
            lambda_value=lam,
            positions_nm=probe_input["positions_nm"],
            particle_masses_dalton=probe_input.get("particle_masses_dalton"),
            box_vectors_nm=probe_input.get("box_vectors_nm"),
            platform_name=args.platform,
        )
        return {
            "ok": True,
            "command": "probe",
            "lambda": result.lambda_value,
            "platform": result.platform_name,
            "energy_kj_mol": result.energy_kj_mol,
            "forces_kj_mol_nm": [
                list(vector) for vector in result.forces_kj_mol_nm
            ],
            "max_force_norm_kj_mol_nm": (
                result.max_force_norm_kj_mol_nm
            ),
            "xml_roundtrip": bool(args.xml_roundtrip),
            "force_xml_sha256": xml_sha256,
            "protocol_sha256": controller.protocol_sha256(lambdas=[lam]),
        }

    if args.command == "benchmark":
        benchmark_input = _cli_read_json_mapping(args.input, "benchmark")
        frames = benchmark_input.get("frames_nm")
        if frames is None and "positions_nm" in benchmark_input:
            frames = [benchmark_input["positions_nm"]]
        if frames is None:
            raise NeuralPathConfigError(
                "benchmark 输入缺少 frames_nm 或 positions_nm"
            )
        report = benchmark_torchforce_outer_lambda(
            controller,
            lambdas=args.lambdas,
            frames_nm=frames,
            particle_masses_dalton=benchmark_input.get(
                "particle_masses_dalton"
            ),
            box_vectors_nm=benchmark_input.get("box_vectors_nm"),
            platform_name=args.platform,
        )
        report["ok"] = True
        report["command"] = "benchmark"
        return report

    if args.command == "nvt-smoke":
        nvt_input = _cli_read_json_mapping(args.input, "nvt-smoke")
        if "positions_nm" not in nvt_input:
            raise NeuralPathConfigError("nvt-smoke 输入缺少 positions_nm")
        report = run_torchforce_nvt_smoke(
            controller,
            lambda_value=args.lambda_value,
            positions_nm=nvt_input["positions_nm"],
            n_steps=args.steps,
            report_interval=args.report_interval,
            timestep_fs=args.timestep_fs,
            temperature_kelvin=args.temperature_k,
            friction_per_ps=args.friction_per_ps,
            particle_masses_dalton=nvt_input.get(
                "particle_masses_dalton"
            ),
            box_vectors_nm=nvt_input.get("box_vectors_nm"),
            platform_name=args.platform,
            random_seed=args.seed,
        )
        report["ok"] = True
        report["command"] = "nvt-smoke"
        return report

    if args.command == "label-existing":
        existing_input = _cli_read_json_mapping(
            args.input, "label-existing"
        )
        frames = existing_input.get("frames_nm")
        if frames is None and "positions_nm" in existing_input:
            frames = [existing_input["positions_nm"]]
        required_fields = (
            "ligand_indices",
            "environment_indices",
            "atomic_numbers",
        )
        missing = [
            field for field in required_fields if field not in existing_input
        ]
        if frames is None or missing:
            raise NeuralPathConfigError(
                "label-existing 输入缺少: "
                + ", ".join(
                    (["frames_nm/positions_nm"] if frames is None else [])
                    + missing
                )
            )
        report = benchmark_existing_orb_mace_basis(
            controller,
            model_name=args.model_name,
            device=args.device,
            lambdas=args.lambdas,
            frames_nm=frames,
            ligand_indices=existing_input["ligand_indices"],
            environment_indices=existing_input["environment_indices"],
            atomic_numbers=existing_input["atomic_numbers"],
        )
        report["ok"] = True
        report["command"] = "label-existing"
        return report

    if args.command == "label-trajectory":
        try:
            import mdtraj as md
        except ImportError as exc:
            raise NeuralPathConfigError(
                "label-trajectory 需要安装 mdtraj"
            ) from exc
        trajectory_path = Path(args.trajectory).expanduser()
        topology_path = Path(args.topology).expanduser()
        if not trajectory_path.is_file() or not topology_path.is_file():
            raise NeuralPathConfigError("trajectory/topology 文件不存在")
        try:
            with md.open(str(trajectory_path)) as trajectory_handle:
                n_trajectory_frames = len(trajectory_handle)
        except Exception as exc:
            raise NeuralPathConfigError(
                f"无法读取轨迹 frame 数: {exc}"
            ) from exc
        frame_indices = _resolve_trajectory_frame_spec(
            args.frames, n_trajectory_frames
        )
        loaded_frames = [
            md.load_frame(
                str(trajectory_path),
                frame_index,
                top=str(topology_path),
            )
            for frame_index in frame_indices
        ]
        frames_nm = [frame.xyz[0].tolist() for frame in loaded_frames]
        topology = loaded_frames[0].topology
        atomic_numbers = []
        for atom in topology.atoms:
            if atom.element is None:
                raise NeuralPathConfigError(
                    f"topology atom {atom.index} 缺少元素"
                )
            atomic_numbers.append(int(atom.element.atomic_number))
        selection_meta = _cli_read_json_mapping(
            args.selection_meta, "selection-meta"
        )
        if (
            "ligand_indices" not in selection_meta
            or "env_indices" not in selection_meta
        ):
            raise NeuralPathConfigError(
                "selection meta 缺少 ligand_indices/env_indices"
            )
        report = benchmark_existing_orb_mace_basis(
            controller,
            model_name=args.model_name,
            device=args.device,
            lambdas=args.lambdas,
            frames_nm=frames_nm,
            ligand_indices=selection_meta["ligand_indices"],
            environment_indices=selection_meta["env_indices"],
            atomic_numbers=atomic_numbers,
        )
        report["ok"] = True
        report["command"] = "label-trajectory"
        report["trajectory"] = str(trajectory_path.resolve())
        report["topology"] = str(topology_path.resolve())
        report["trajectory_frame_count"] = n_trajectory_frames
        report["frame_indices"] = list(frame_indices)
        return report

    if args.command == "mace-nvt-smoke":
        try:
            import mdtraj as md
        except ImportError as exc:
            raise NeuralPathConfigError(
                "mace-nvt-smoke 需要安装 mdtraj"
            ) from exc
        system_xml_path = Path(args.system_xml).expanduser()
        trajectory_path = Path(args.trajectory).expanduser()
        topology_path = Path(args.topology).expanduser()
        for label, path in (
            ("system XML", system_xml_path),
            ("trajectory", trajectory_path),
            ("topology", topology_path),
        ):
            if not path.is_file():
                raise NeuralPathConfigError(f"{label} 文件不存在: {path}")
        try:
            with md.open(str(trajectory_path)) as trajectory_handle:
                trajectory_frame_count = len(trajectory_handle)
            frame_indices = _resolve_trajectory_frame_spec(
                args.frame, trajectory_frame_count
            )
            if len(frame_indices) != 1:
                raise NeuralPathConfigError(
                    "mace-nvt-smoke --frame 必须只选择一个 frame"
                )
            frame = md.load_frame(
                str(trajectory_path),
                frame_indices[0],
                top=str(topology_path),
            )
        except NeuralPathConfigError:
            raise
        except Exception as exc:
            raise NeuralPathConfigError(
                f"mace-nvt-smoke 无法读取轨迹: {exc}"
            ) from exc
        atomic_numbers = []
        for atom in frame.topology.atoms:
            if atom.element is None:
                raise NeuralPathConfigError(
                    f"topology atom {atom.index} 缺少元素"
                )
            atomic_numbers.append(int(atom.element.atomic_number))
        selection_meta = _cli_read_json_mapping(
            args.selection_meta, "selection-meta"
        )
        try:
            openmm, _ = _require_openmm()
            base_system = openmm.XmlSerializer.deserialize(
                system_xml_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise TorchForceDeploymentError(
                f"base System XML 反序列化失败: {exc}"
            ) from exc
        if frame.unitcell_vectors is None:
            raise NeuralPathConfigError("轨迹 frame 缺少周期盒向量")
        report = run_mace_decomposition_nvt_smoke(
            controller,
            base_system,
            atomic_numbers=atomic_numbers,
            ligand_indices=selection_meta.get("ligand_indices"),
            environment_indices=selection_meta.get("env_indices"),
            positions_nm=frame.xyz[0].tolist(),
            box_vectors_nm=frame.unitcell_vectors[0].tolist(),
            lambda_value=args.lambda_value,
            n_steps=args.steps,
            report_interval=args.report_interval,
            timestep_fs=args.timestep_fs,
            temperature_kelvin=args.temperature_k,
            friction_per_ps=args.friction_per_ps,
            device=args.device,
            platform_name=args.platform,
            random_seed=args.seed,
        )
        report["ok"] = True
        report["command"] = "mace-nvt-smoke"
        report["system_xml"] = str(system_xml_path.resolve())
        report["trajectory"] = str(trajectory_path.resolve())
        report["topology"] = str(topology_path.resolve())
        report["frame_index"] = frame_indices[0]
        return report

    if args.command == "compare":
        comparison_input = _cli_read_json_mapping(args.input, "compare")
        arms = comparison_input.get("arms")
        if (
            not isinstance(arms, Sequence)
            or isinstance(arms, (str, bytes))
        ):
            raise NeuralPathConfigError("compare 输入的 arms 必须是序列")
        thresholds = comparison_input.get("thresholds", {})
        if not isinstance(thresholds, Mapping):
            raise NeuralPathConfigError("compare thresholds 必须是对象")
        allowed_thresholds = {
            "delta_g_sigma_multiplier",
            "minimum_ess_gpu_improvement_fraction",
            "minimum_unique_gain_over_relayout_fraction",
            "anomaly_rate_tolerance",
        }
        unknown_thresholds = set(thresholds) - allowed_thresholds
        if unknown_thresholds:
            raise NeuralPathConfigError(
                "compare 含未知 thresholds: "
                + ", ".join(sorted(map(str, unknown_thresholds)))
            )
        report = compare_wp5_arms(arms, **dict(thresholds))
        report["ok"] = True
        report["command"] = "compare"
        report["protocol_sha256"] = controller.protocol_sha256()
        return report

    if args.command == "compare-replicates":
        comparison_input = _cli_read_json_mapping(
            args.input, "compare-replicates"
        )
        arms = comparison_input.get("arms")
        if (
            not isinstance(arms, Sequence)
            or isinstance(arms, (str, bytes))
        ):
            raise NeuralPathConfigError(
                "compare-replicates 输入的 arms 必须是序列"
            )
        thresholds = comparison_input.get("thresholds", {})
        if not isinstance(thresholds, Mapping):
            raise NeuralPathConfigError(
                "compare-replicates thresholds 必须是对象"
            )
        allowed_thresholds = {
            "delta_g_sigma_multiplier",
            "minimum_ess_gpu_improvement_fraction",
            "minimum_unique_gain_over_relayout_fraction",
            "anomaly_rate_tolerance",
        }
        unknown_thresholds = set(thresholds) - allowed_thresholds
        if unknown_thresholds:
            raise NeuralPathConfigError(
                "compare-replicates 含未知 thresholds: "
                + ", ".join(sorted(map(str, unknown_thresholds)))
            )
        report = compare_wp5_replicates(
            arms,
            minimum_replicates=args.minimum_replicates,
            **dict(thresholds),
        )
        report["ok"] = True
        report["command"] = "compare-replicates"
        report["protocol_sha256"] = controller.protocol_sha256()
        return report

    if args.command == "qualify":
        manifest_payload = _cli_read_json_mapping(args.manifest, "manifest")
        manifest = NeuralBasisTaskManifest.from_mapping(
            manifest_payload, verify_training_data=True
        )
        benchmark_report = _cli_read_json_mapping(
            args.benchmark_report, "benchmark-report"
        )
        nvt_report = _cli_read_json_mapping(args.nvt_report, "nvt-report")
        report = qualify_wp4_basis(
            controller,
            manifest,
            benchmark_report,
            nvt_report,
            max_seconds_per_frame=args.max_seconds_per_frame,
        )
        report["ok"] = True
        report["command"] = "qualify"
        return report

    raise NeuralPathConfigError(f"未知 CLI command: {args.command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """独立 CLI 入口；返回 shell exit code，不调用任何 ABFE 主程序。"""

    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = _run_cli_command(args)
        _cli_write_json(result, args.output)
        return 0
    except (
        NeuralPathConfigError,
        NeuralPathIntegrityError,
        NeuralPathFrameError,
        TorchForceDeploymentError,
        OSError,
    ) as exc:
        error = {
            "ok": False,
            "command": getattr(args, "command", None),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(
            json.dumps(
                error,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


__all__ = [
    "NEURAL_PATH_PROTOCOL_VERSION",
    "NEURAL_BASIS_MODEL_PROTOCOL_VERSION",
    "NEURAL_PATH_ACCOUNTING_VERSION",
    "NeuralPathConfigError",
    "NeuralPathIntegrityError",
    "NeuralPathFrameError",
    "TorchForceDeploymentError",
    "NeuralBasisSupportDomain",
    "SupportDomainEvaluation",
    "NeuralBasisModelSpec",
    "NeuralBasisTaskManifest",
    "NeuralPathSafety",
    "OuterLambdaController",
    "AnalyticBasisEvaluation",
    "HarmonicDistanceBasis",
    "ExistingOpenMMMLBasisEvaluation",
    "ExistingOrbMaceBasisAdapter",
    "MaceDecompositionPythonComputation",
    "IBSEnergyFrame",
    "IBSEnergyLedger",
    "compose_ibs_energy_frame",
    "OpenMMPathEvaluation",
    "build_torchforce_from_spec",
    "build_openmm_outer_lambda_force",
    "build_torchforce_outer_lambda_force",
    "build_mace_decomposition_python_force",
    "evaluate_outer_lambda_force_group_states",
    "run_mace_decomposition_nvt_smoke",
    "serialize_openmm_force",
    "deserialize_openmm_force",
    "evaluate_openmm_outer_lambda_force",
    "summarize_finite_series",
    "benchmark_torchforce_outer_lambda",
    "benchmark_existing_orb_mace_basis",
    "prepare_existing_model_node_config",
    "resolve_existing_model_artifact",
    "run_torchforce_nvt_smoke",
    "qualify_wp4_basis",
    "importance_effective_sample_size",
    "count_discrete_transitions",
    "integrated_autocorrelation_time",
    "analyze_wp5_arm",
    "compare_wp5_arms",
    "compare_wp5_replicates",
    "load_neural_path_config",
    "sha256_file",
    "stable_payload_sha256",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())

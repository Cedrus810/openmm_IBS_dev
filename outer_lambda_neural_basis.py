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
    if isinstance(value, Path):
        path = value.expanduser()
        raw = str(value)
    else:
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
    coordinate_imaging: str = "none"
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
        coordinate_imaging = _nonempty_string(
            config.get(
                "coordinate_imaging",
                (
                    "minimum_image_local"
                    if backend == "existing_openmmml"
                    else "none"
                ),
            ),
            f"basis[{name}].coordinate_imaging",
        )
        if coordinate_imaging not in {"none", "minimum_image_local"}:
            raise NeuralPathConfigError(
                f"basis[{name}].coordinate_imaging 必须是 "
                "'none' 或 'minimum_image_local'"
            )
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
            coordinate_imaging=coordinate_imaging,
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
            "coordinate_imaging": self.coordinate_imaging,
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


def run_mace_decomposition_nvt(
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
    """在 base System 深拷贝中运行真实 MACE 路径 NVT，不修改调用方 System。"""

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
        raise NeuralPathConfigError("MACE NVT 必须提供周期盒")
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
    # NVT 明确移除复制体中的 barostat；原 System 不受影响。
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
    integration_seconds = 0.0
    diagnostic_seconds = 0.0
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
            integration_started = time.perf_counter()
            integrator.step(chunk)
            integration_seconds += time.perf_counter() - integration_started
            completed += chunk
            diagnostic_started = time.perf_counter()
            base_state = context.getState(
                getEnergy=True, groups=base_mask
            )
            path_state = context.getState(
                getEnergy=True, getForces=True, groups=path_mask
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
            path_forces = path_state.getForces(
                asNumpy=True
            ).value_in_unit(
                unit.kilojoules_per_mole / unit.nanometer
            )
            force_norms = [
                math.sqrt(
                    math.fsum(float(component) ** 2 for component in vector)
                )
                for vector in forces
            ]
            path_force_norms = [
                math.sqrt(
                    math.fsum(float(component) ** 2 for component in vector)
                )
                for vector in path_forces
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
                and all(math.isfinite(value) for value in path_force_norms)
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
                    "max_path_force_kj_mol_nm": max(
                        path_force_norms, default=0.0
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
            diagnostic_seconds += time.perf_counter() - diagnostic_started
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
            f"MACE decomposition NVT 失败: {exc}"
        ) from exc
    finally:
        if "context" in locals():
            del context
        del integrator
    elapsed = time.perf_counter() - started
    support_violation_count = sum(
        1
        for sample in samples
        if any(
            not evaluation["supported"]
            for evaluation in sample["support_domain"]
        )
    )
    fail_on_support = (
        controller.safety is not None
        and controller.safety.fail_on_support_domain_violation
    )
    return {
        "report_type": "outer_lambda_mace_decomposition_nvt",
        "report_version": 1,
        "passed": not (support_violation_count and fail_on_support),
        "platform": platform_name,
        "device": device,
        "lambda": lam,
        "path_force_group": path_force_group,
        "n_steps": n_steps,
        "report_interval": report_interval,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / n_steps,
        "integration_seconds": integration_seconds,
        "integration_seconds_per_step": integration_seconds / n_steps,
        "diagnostic_seconds": diagnostic_seconds,
        "diagnostic_seconds_per_report": (
            diagnostic_seconds / len(samples) if samples else None
        ),
        "support_domain_configured": all(
            basis.support_domain is not None for basis in controller.bases
        ),
        "support_domain_violation_count": support_violation_count,
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
        "max_path_force_kj_mol_nm": summarize_finite_series(
            [sample["max_path_force_kj_mol_nm"] for sample in samples]
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


def run_mace_decomposition_nvt_smoke(*args, **kwargs) -> dict[str, Any]:
    """兼容旧调用的快速连通性包装；正式验收使用通用 NVT 执行器。"""

    report = run_mace_decomposition_nvt(*args, **kwargs)
    report["report_type"] = "outer_lambda_mace_decomposition_nvt_smoke"
    return report


def _openmm_system_degrees_of_freedom(system: Any) -> int:
    """计算恒温诊断使用的经典自由度。"""

    openmm, unit = _require_openmm()
    positive_mass_particles = sum(
        1
        for index in range(system.getNumParticles())
        if system.getParticleMass(index).value_in_unit(unit.dalton) > 0.0
    )
    dof = 3 * positive_mass_particles - system.getNumConstraints()
    if any(
        isinstance(system.getForce(index), openmm.CMMotionRemover)
        for index in range(system.getNumForces())
    ):
        dof -= 3
    if dof <= 0:
        raise NeuralPathConfigError("OpenMM System 的有效自由度必须为正")
    return dof


def _aligned_selected_rmsd_nm(
    positions_nm: Sequence[Sequence[float]],
    reference_positions_nm: Sequence[Sequence[float]],
    atom_indices: Sequence[int],
    box_vectors_nm: Sequence[Sequence[float]],
) -> float:
    """minimum-image 后对固定选择做 Kabsch 对齐 RMSD。"""

    import numpy as np

    current = np.asarray(positions_nm, dtype=np.float64)
    reference = np.asarray(reference_positions_nm, dtype=np.float64)
    indices = list(atom_indices)
    box = np.asarray(_normalize_box_vectors_nm(box_vectors_nm), dtype=np.float64)
    inverse_box = np.linalg.inv(box)

    def image_selected(frame):
        selected = frame[indices].copy()
        anchor = selected[0].copy()
        displacement = selected - anchor
        fractional = displacement @ inverse_box
        displacement -= np.floor(fractional + 0.5) @ box
        return anchor + displacement

    mobile = image_selected(current)
    target = image_selected(reference)
    mobile -= np.mean(mobile, axis=0)
    target -= np.mean(target, axis=0)
    covariance = mobile.T @ target
    left, _, right = np.linalg.svd(covariance)
    correction = np.identity(3)
    correction[2, 2] = np.sign(np.linalg.det(left @ right))
    rotation = left @ correction @ right
    difference = mobile @ rotation - target
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))


def run_mace_decomposition_mts_arm(
    controller: OuterLambdaController,
    base_system: Any,
    *,
    atomic_numbers: Sequence[int],
    ligand_indices: Sequence[int],
    environment_indices: Sequence[int],
    positions_nm: Sequence[Sequence[float]],
    box_vectors_nm: Sequence[Sequence[float]],
    torsion_atom_indices: Sequence[int],
    mts_ratio: int,
    lambda_value: float = 0.5,
    n_inner_steps: int = 10_000,
    report_interval_inner_steps: int = 100,
    inner_timestep_fs: float = 0.5,
    temperature_kelvin: float = 300.0,
    friction_per_ps: float = 1.0,
    device: str = "cuda",
    platform_name: str = "CUDA",
    random_seed: int = 20260730,
    required_coefficient: float = 0.09,
) -> dict[str, Any]:
    """运行 EXP-009 单个 BAOAB-rRESPA MTS ratio，固定 MACE 为慢力。"""

    import numpy as np

    if (
        isinstance(mts_ratio, bool)
        or not isinstance(mts_ratio, int)
        or mts_ratio not in {1, 2, 4, 8}
    ):
        raise NeuralPathConfigError("mts_ratio 必须是 1/2/4/8")
    if controller.basis_count != 1 or not controller.enabled:
        raise NeuralPathConfigError("EXP-009 要求 enabled 且 M=1")
    frozen_coefficient = _finite_float(
        required_coefficient, "required_coefficient"
    )
    if controller.coefficients != (frozen_coefficient,):
        raise NeuralPathConfigError(
            "EXP-009 coefficient 已冻结为 "
            f"{frozen_coefficient}，配置为 {controller.coefficients}"
        )
    for field, value in (
        ("n_inner_steps", n_inner_steps),
        ("report_interval_inner_steps", report_interval_inner_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise NeuralPathConfigError(f"{field} 必须是正整数")
        if value % mts_ratio != 0:
            raise NeuralPathConfigError(
                f"{field} 必须能被 mts_ratio={mts_ratio} 整除"
            )
    lam = _finite_float(lambda_value, "lambda_value")
    controller.envelope(lam)
    inner_dt = _finite_float(inner_timestep_fs, "inner_timestep_fs")
    temperature = _finite_float(temperature_kelvin, "temperature_kelvin")
    friction = _finite_float(friction_per_ps, "friction_per_ps")
    if min(inner_dt, temperature, friction) <= 0.0:
        raise NeuralPathConfigError("MTS 时间步、温度和摩擦必须为正")
    normalized_positions = _normalize_frame_collection([positions_nm])[0]
    box_rows = _normalize_box_vectors_nm(box_vectors_nm)
    if box_rows is None:
        raise NeuralPathConfigError("MACE MTS 必须提供周期盒")
    initial_support = controller.evaluate_support_domains(
        normalized_positions,
        box_vectors_nm=box_rows,
    )
    initial_support_payload = [
        evaluation.payload() for evaluation in initial_support
    ]
    initial_unsupported = [
        evaluation
        for evaluation in initial_support
        if not evaluation.supported
    ]
    if (
        initial_unsupported
        and controller.safety is not None
        and controller.safety.fail_on_support_domain_violation
    ):
        details = "; ".join(
            f"{evaluation.basis_name}: violations="
            f"{','.join(evaluation.violations)}, "
            f"max_pair={evaluation.max_pair_distance_nm!r} nm, "
            f"Rg={evaluation.radius_of_gyration_nm:.9g} nm"
            for evaluation in initial_unsupported
        )
        raise NeuralPathFrameError(
            "MACE MTS 初始坐标不在冻结支持域，积分未启动: " + details
        )
    openmm, unit = _require_openmm()
    try:
        system = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(base_system)
        )
    except Exception as exc:
        raise TorchForceDeploymentError(f"无法复制 base System: {exc}") from exc
    if system.getNumParticles() != len(normalized_positions):
        raise NeuralPathConfigError("base System 粒子数与 positions 不一致")
    for force_index in reversed(range(system.getNumForces())):
        force = system.getForce(force_index)
        if "Barostat" in type(force).__name__:
            system.removeForce(force_index)
            continue
        force.setForceGroup(0)
        if hasattr(force, "setReciprocalSpaceForceGroup"):
            force.setReciprocalSpaceForceGroup(0)
    path_group = 31
    path_force = build_mace_decomposition_python_force(
        controller,
        atomic_numbers=atomic_numbers,
        ligand_indices=ligand_indices,
        environment_indices=environment_indices,
        default_lambda=lam,
        device=device,
        force_group=path_group,
    )
    if (
        platform_name.split(":", 1)[0].strip().upper() == "CUDA"
        and device.lower() == "cuda"
        and type(path_force).__name__ == "PythonForce"
    ):
        raise TorchForceDeploymentError(
            "EXP-009 后端不兼容: 当前完整 MACE 路径是 "
            "openmm.PythonForce，节点实测它在 OpenMM CUDA "
            "MTSLangevinIntegrator force-group 内核中触发 "
            "CUDA_ERROR_INVALID_HANDLE。普通 LangevinMiddleIntegrator "
            "通过不能证明该组合可用于 MTS。禁止继续重试或调 coefficient；"
            "预注册决策为 start_exp010_cheap_cv_due_to_backend。"
        )
    system.addForce(path_force)
    vectors = tuple(
        openmm.Vec3(*row) * unit.nanometer for row in box_rows
    )
    system.setDefaultPeriodicBoxVectors(*vectors)
    outer_timestep_fs = mts_ratio * inner_dt
    integrator = openmm.MTSLangevinIntegrator(
        temperature * unit.kelvin,
        friction / unit.picosecond,
        outer_timestep_fs * unit.femtoseconds,
        [(path_group, 1), (0, mts_ratio)],
    )
    integrator.setRandomNumberSeed(int(random_seed))
    platform = openmm.Platform.getPlatformByName(platform_name)
    dof = _openmm_system_degrees_of_freedom(system)
    gas_constant = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
        unit.kilojoules_per_mole / unit.kelvin
    )
    reference_positions = np.asarray(normalized_positions, dtype=np.float64)
    samples = []
    integration_seconds = 0.0
    diagnostic_seconds = 0.0
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
        completed_inner = 0
        outer_chunk = report_interval_inner_steps // mts_ratio
        path_mask = 1 << path_group
        base_mask = 1
        while completed_inner < n_inner_steps:
            integration_started = time.perf_counter()
            integrator.step(outer_chunk)
            integration_seconds += time.perf_counter() - integration_started
            completed_inner += report_interval_inner_steps
            diagnostic_started = time.perf_counter()
            base_state = context.getState(getEnergy=True, groups=base_mask)
            path_state = context.getState(
                getEnergy=True, getForces=True, groups=path_mask
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
            kinetic_energy = total_state.getKineticEnergy().value_in_unit(
                unit.kilojoules_per_mole
            )
            instantaneous_temperature = (
                2.0 * kinetic_energy / (dof * gas_constant)
            )
            path_forces = path_state.getForces(
                asNumpy=True
            ).value_in_unit(
                unit.kilojoules_per_mole / unit.nanometer
            )
            path_force_norms = np.linalg.norm(
                np.asarray(path_forces, dtype=np.float64), axis=1
            )
            current_positions = total_state.getPositions(
                asNumpy=True
            ).value_in_unit(unit.nanometer)
            current_positions_array = np.asarray(
                current_positions, dtype=np.float64
            )
            support = controller.evaluate_support_domains(
                current_positions_array.tolist(),
                box_vectors_nm=box_rows,
            )
            sample = {
                "inner_step": completed_inner,
                "physical_time_ps": completed_inner * inner_dt / 1000.0,
                "base_energy_kj_mol": float(base_energy),
                "path_energy_kj_mol": float(path_energy),
                "total_energy_kj_mol": float(total_energy),
                "kinetic_energy_kj_mol": float(kinetic_energy),
                "temperature_kelvin": float(instantaneous_temperature),
                "energy_closure_error_kj_mol": float(
                    total_energy - base_energy - path_energy
                ),
                "max_path_force_kj_mol_nm": float(
                    np.max(path_force_norms)
                    if path_force_norms.size
                    else 0.0
                ),
                "slow_torsion_degrees": periodic_dihedral_degrees(
                    current_positions_array.tolist(),
                    torsion_atom_indices,
                    box_vectors_nm=box_rows,
                ),
                "selected_aligned_rmsd_nm": _aligned_selected_rmsd_nm(
                    current_positions_array,
                    reference_positions,
                    ligand_indices,
                    box_rows,
                ),
                "support_domain": [
                    evaluation.payload() for evaluation in support
                ],
            }
            if any(
                not math.isfinite(float(value))
                for key, value in sample.items()
                if key
                not in {
                    "support_domain",
                    "inner_step",
                }
            ):
                raise TorchForceDeploymentError(
                    f"MACE MTS N={mts_ratio} 出现非有限诊断"
                )
            samples.append(sample)
            diagnostic_seconds += time.perf_counter() - diagnostic_started
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
            f"MACE MTS N={mts_ratio} 失败: {exc}"
        ) from exc
    finally:
        if "context" in locals():
            del context
        del integrator
    elapsed = time.perf_counter() - started
    support_violations = sum(
        1
        for sample in samples
        if any(
            not evaluation["supported"]
            for evaluation in sample["support_domain"]
        )
    )
    simulated_ns = n_inner_steps * inner_dt * 1.0e-6
    return {
        "report_type": "outer_lambda_mace_mts_arm",
        "report_version": 1,
        "passed": support_violations == 0,
        "mts_ratio": mts_ratio,
        "inner_timestep_fs": inner_dt,
        "outer_timestep_fs": outer_timestep_fs,
        "mace_interval_fs": outer_timestep_fs,
        "n_inner_steps": n_inner_steps,
        "n_outer_steps": n_inner_steps // mts_ratio,
        "simulated_time_ps": n_inner_steps * inner_dt / 1000.0,
        "report_interval_inner_steps": report_interval_inner_steps,
        "temperature_target_kelvin": temperature,
        "lambda": lam,
        "coefficient": controller.coefficients[0],
        "random_seed": int(random_seed),
        "platform": platform_name,
        "device": device,
        "force_groups": {
            "base": 0,
            "mace_path": path_group,
            "mts_groups": [[path_group, 1], [0, mts_ratio]],
        },
        "elapsed_seconds": elapsed,
        "integration_seconds": integration_seconds,
        "diagnostic_seconds": diagnostic_seconds,
        "integration_seconds_per_inner_step": (
            integration_seconds / n_inner_steps
        ),
        "ns_per_day": (
            simulated_ns * 86400.0 / integration_seconds
            if integration_seconds > 0.0
            else None
        ),
        "support_domain_configured": all(
            basis.support_domain is not None for basis in controller.bases
        ),
        "initial_support_domain": initial_support_payload,
        "support_domain_violation_count": support_violations,
        "max_energy_closure_error_kj_mol": max(
            (
                abs(sample["energy_closure_error_kj_mol"])
                for sample in samples
            ),
            default=0.0,
        ),
        "base_energy_kj_mol": summarize_finite_series(
            [sample["base_energy_kj_mol"] for sample in samples]
        ),
        "path_energy_kj_mol": summarize_finite_series(
            [sample["path_energy_kj_mol"] for sample in samples]
        ),
        "total_energy_kj_mol": summarize_finite_series(
            [sample["total_energy_kj_mol"] for sample in samples]
        ),
        "temperature_kelvin": summarize_finite_series(
            [sample["temperature_kelvin"] for sample in samples]
        ),
        "max_path_force_kj_mol_nm": summarize_finite_series(
            [sample["max_path_force_kj_mol_nm"] for sample in samples]
        ),
        "selected_aligned_rmsd_nm": summarize_finite_series(
            [sample["selected_aligned_rmsd_nm"] for sample in samples]
        ),
        "slow_torsion": analyze_periodic_torsion_series(
            [sample["slow_torsion_degrees"] for sample in samples]
        ),
        "samples": samples,
        "protocol_sha256": controller.protocol_sha256(lambdas=[lam]),
    }


def _jensen_shannon_divergence(
    probabilities_a: Sequence[float],
    probabilities_b: Sequence[float],
) -> float:
    """自然对数定义的有限 Jensen-Shannon divergence。"""

    a = tuple(_finite_float(value, "probability_a") for value in probabilities_a)
    b = tuple(_finite_float(value, "probability_b") for value in probabilities_b)
    if len(a) != len(b) or not a:
        raise NeuralPathConfigError("概率向量必须等长且非空")
    if any(value < 0.0 for value in a + b):
        raise NeuralPathConfigError("概率不允许为负")
    sum_a = math.fsum(a)
    sum_b = math.fsum(b)
    if min(sum_a, sum_b) <= 0.0:
        raise NeuralPathConfigError("概率向量总和必须为正")
    normalized_a = tuple(value / sum_a for value in a)
    normalized_b = tuple(value / sum_b for value in b)
    midpoint = tuple(
        0.5 * (left + right)
        for left, right in zip(normalized_a, normalized_b, strict=True)
    )

    def divergence(values):
        return math.fsum(
            value * math.log(value / center)
            for value, center in zip(values, midpoint, strict=True)
            if value > 0.0
        )

    return 0.5 * (divergence(normalized_a) + divergence(normalized_b))


def assess_mace_mts_matrix(
    arm_reports: Sequence[Mapping[str, Any]],
    *,
    max_path_force_kj_mol_nm: float = 250.0,
    max_energy_closure_error_kj_mol: float = 0.1,
    max_temperature_mean_difference_kelvin: float = 5.0,
    max_energy_standardized_mean_difference: float = 0.25,
    max_torsion_js_divergence: float = 0.05,
    max_rmsd_mean_difference_nm: float = 0.05,
    minimum_n4_ns_per_day: float = 1.0,
) -> dict[str, Any]:
    """以 N=1 为参考执行 EXP-009 的 N=1/2/4 分布和性能硬门。"""

    reports = {int(report.get("mts_ratio", -1)): report for report in arm_reports}
    if set(reports) != {1, 2, 4} or len(arm_reports) != 3:
        raise NeuralPathConfigError("MTS matrix 必须恰好包含 N=1/2/4")
    limits = {
        "max_path_force_kj_mol_nm": _finite_float(
            max_path_force_kj_mol_nm, "max_path_force_kj_mol_nm"
        ),
        "max_energy_closure_error_kj_mol": _finite_float(
            max_energy_closure_error_kj_mol,
            "max_energy_closure_error_kj_mol",
        ),
        "max_temperature_mean_difference_kelvin": _finite_float(
            max_temperature_mean_difference_kelvin,
            "max_temperature_mean_difference_kelvin",
        ),
        "max_energy_standardized_mean_difference": _finite_float(
            max_energy_standardized_mean_difference,
            "max_energy_standardized_mean_difference",
        ),
        "max_torsion_js_divergence": _finite_float(
            max_torsion_js_divergence, "max_torsion_js_divergence"
        ),
        "max_rmsd_mean_difference_nm": _finite_float(
            max_rmsd_mean_difference_nm,
            "max_rmsd_mean_difference_nm",
        ),
        "minimum_n4_ns_per_day": _finite_float(
            minimum_n4_ns_per_day, "minimum_n4_ns_per_day"
        ),
    }
    if min(limits.values()) <= 0.0:
        raise NeuralPathConfigError("所有 MTS qualification 阈值必须为正")
    reference = reports[1]
    reference_energy = reference["total_energy_kj_mol"]
    reference_temperature = reference["temperature_kelvin"]
    reference_rmsd = reference["selected_aligned_rmsd_nm"]
    reference_histogram = reference["slow_torsion"]["histogram"][
        "probabilities"
    ]
    arm_assessments = {}
    for ratio in (1, 2, 4):
        report = reports[ratio]
        energy_scale = max(
            float(reference_energy["std"]),
            float(report["total_energy_kj_mol"]["std"]),
            1.0e-12,
        )
        energy_effect = abs(
            float(report["total_energy_kj_mol"]["mean"])
            - float(reference_energy["mean"])
        ) / energy_scale
        temperature_difference = abs(
            float(report["temperature_kelvin"]["mean"])
            - float(reference_temperature["mean"])
        )
        rmsd_difference = abs(
            float(report["selected_aligned_rmsd_nm"]["mean"])
            - float(reference_rmsd["mean"])
        )
        torsion_js = _jensen_shannon_divergence(
            report["slow_torsion"]["histogram"]["probabilities"],
            reference_histogram,
        )
        checks = {
            "run_completed": report.get("passed") is True,
            "support_domain": (
                report.get("support_domain_configured") is True
                and report.get("support_domain_violation_count") == 0
            ),
            "path_force": (
                float(report["max_path_force_kj_mol_nm"]["max"])
                <= limits["max_path_force_kj_mol_nm"]
            ),
            "energy_closure": (
                float(report["max_energy_closure_error_kj_mol"])
                <= limits["max_energy_closure_error_kj_mol"]
            ),
            "temperature_distribution": (
                temperature_difference
                <= limits["max_temperature_mean_difference_kelvin"]
            ),
            "energy_distribution": (
                energy_effect
                <= limits["max_energy_standardized_mean_difference"]
            ),
            "torsion_distribution": (
                torsion_js <= limits["max_torsion_js_divergence"]
            ),
            "structure_distribution": (
                rmsd_difference <= limits["max_rmsd_mean_difference_nm"]
            ),
        }
        arm_assessments[str(ratio)] = {
            "passed": all(checks.values()),
            "checks": checks,
            "observed_vs_n1": {
                "temperature_mean_difference_kelvin": temperature_difference,
                "energy_standardized_mean_difference": energy_effect,
                "torsion_js_divergence": torsion_js,
                "rmsd_mean_difference_nm": rmsd_difference,
            },
            "ns_per_day": report.get("ns_per_day"),
        }
    physics_passed = all(
        arm_assessments[str(ratio)]["passed"] for ratio in (1, 2, 4)
    )
    speed_passed = (
        float(reports[4].get("ns_per_day", 0.0))
        >= limits["minimum_n4_ns_per_day"]
    )
    if not physics_passed:
        decision = "direct_mace_teacher_only_due_to_mts_bias"
    elif not speed_passed:
        decision = "start_exp010_cheap_cv_due_to_cost"
    else:
        decision = "qualified_to_test_n8_before_wp5"
    return {
        "report_type": "outer_lambda_mace_mts_qualification",
        "report_version": 1,
        "qualified": physics_passed and speed_passed,
        "physics_passed": physics_passed,
        "speed_passed": speed_passed,
        "decision": decision,
        "thresholds": limits,
        "arms": arm_assessments,
        "arm_reports": [dict(reports[ratio]) for ratio in (1, 2, 4)],
    }


def assess_mace_nvt_qualification(
    run_report: Mapping[str, Any],
    *,
    minimum_steps: int = 1000,
    max_path_force_kj_mol_nm: float = 250.0,
    max_energy_closure_error_kj_mol: float = 0.1,
    max_integration_seconds_per_step: float = 0.2,
) -> dict[str, Any]:
    """对冻结阈值下的 MACE NVT run 执行 WP-4 qualification 判定。"""

    if not isinstance(run_report, Mapping):
        raise NeuralPathConfigError("run_report 必须是映射")
    if (
        isinstance(minimum_steps, bool)
        or not isinstance(minimum_steps, int)
        or minimum_steps <= 0
    ):
        raise NeuralPathConfigError("minimum_steps 必须是正整数")
    force_limit = _finite_float(
        max_path_force_kj_mol_nm, "max_path_force_kj_mol_nm"
    )
    closure_limit = _finite_float(
        max_energy_closure_error_kj_mol,
        "max_energy_closure_error_kj_mol",
    )
    speed_limit = _finite_float(
        max_integration_seconds_per_step,
        "max_integration_seconds_per_step",
    )
    if min(force_limit, closure_limit, speed_limit) <= 0.0:
        raise NeuralPathConfigError("qualification 阈值必须为正")
    path_force_summary = run_report.get("max_path_force_kj_mol_nm")
    if not isinstance(path_force_summary, Mapping):
        raise NeuralPathConfigError(
            "run report 缺少 max_path_force_kj_mol_nm summary"
        )
    observed_path_force = _finite_float(
        path_force_summary.get("max"),
        "run.max_path_force_kj_mol_nm.max",
    )
    observed_closure = _finite_float(
        run_report.get("max_energy_closure_error_kj_mol"),
        "run.max_energy_closure_error_kj_mol",
    )
    observed_speed = _finite_float(
        run_report.get("integration_seconds_per_step"),
        "run.integration_seconds_per_step",
    )
    checks = {
        "run_completed": run_report.get("passed") is True,
        "minimum_steps": int(run_report.get("n_steps", 0)) >= minimum_steps,
        "support_domain": (
            run_report.get("support_domain_configured") is True
            and run_report.get("support_domain_violation_count") == 0
        ),
        "path_force": observed_path_force <= force_limit,
        "energy_closure": observed_closure <= closure_limit,
        "integration_speed": observed_speed <= speed_limit,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "report_type": "outer_lambda_mace_nvt_qualification",
        "report_version": 1,
        "qualified": not failed_checks,
        "checks": checks,
        "failed_checks": failed_checks,
        "thresholds": {
            "minimum_steps": minimum_steps,
            "max_path_force_kj_mol_nm": force_limit,
            "max_energy_closure_error_kj_mol": closure_limit,
            "max_integration_seconds_per_step": speed_limit,
        },
        "observed": {
            "n_steps": run_report.get("n_steps"),
            "support_domain_violation_count": run_report.get(
                "support_domain_violation_count"
            ),
            "max_path_force_kj_mol_nm": observed_path_force,
            "max_energy_closure_error_kj_mol": observed_closure,
            "integration_seconds_per_step": observed_speed,
        },
        "run_report": dict(run_report),
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


class IBSSamplerNeuralPathAdapter:
    """在不改 ``ibs_engine.py`` 的情况下扩展现有 IBSSampler 能量收集。

    适配器复用 sampler 的 Context、interaction probe、LRC、TMBAR buffer 和四份
    production history。唯一替换的方法是 ``collect_energies()``，并额外保留
    neural-path/basis history 供协议审计。
    """

    def __init__(
        self,
        sampler: Any,
        controller: OuterLambdaController,
        lambdas: Iterable[float],
        ibs_wrapper: OuterLambdaIBSBiasForce,
    ) -> None:
        if not isinstance(ibs_wrapper, OuterLambdaIBSBiasForce):
            raise TypeError(
                "ibs_wrapper 必须是 OuterLambdaIBSBiasForce"
            )
        self.sampler = sampler
        self.controller = controller
        self.lambdas = tuple(
            _finite_float(value, "lambda") for value in lambdas
        )
        if len(self.lambdas) != int(getattr(sampler, "n_states", -1)):
            raise NeuralPathConfigError(
                "lambda schedule 数量与 sampler.n_states 不一致"
            )
        if ibs_wrapper.lambdas != self.lambdas:
            raise NeuralPathConfigError(
                "sampler adapter 与 IBS wrapper 的 lambda schedule 不一致"
            )
        if ibs_wrapper.controller.protocol_sha256(lambdas=self.lambdas) != (
            controller.protocol_sha256(lambdas=self.lambdas)
        ):
            raise NeuralPathConfigError(
                "sampler adapter 与 IBS wrapper 的神经路径协议不一致"
            )
        required_histories = (
            "energy_buffer",
            "energy_history",
            "bias_history",
            "base_energy_history",
        )
        missing = [
            name for name in required_histories if not hasattr(sampler, name)
        ]
        if missing:
            raise NeuralPathConfigError(
                "sampler 缺少历史字段: " + ", ".join(missing)
            )
        self.ibs_wrapper = ibs_wrapper
        self.neural_path_energy_history: list[tuple[float, ...]] = []
        self.basis_energy_history: list[tuple[float, ...]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.sampler, name)

    def _record_query(self, success: bool, reason: str | None = None) -> None:
        recorder = getattr(self.sampler, "_record_energy_query_result", None)
        if callable(recorder):
            recorder(success, reason)

    def collect_energies(self):
        """收集一帧完整神经 IBS 账本；任一分量失败则不追加任何 history。"""

        import numpy as np

        openmm, unit = _require_openmm()
        del openmm
        self.sampler._energy_query_attempts = int(
            getattr(self.sampler, "_energy_query_attempts", 0)
        ) + 1
        try:
            base_state = self.sampler.context.getState(
                getEnergy=True, groups={0, 2, 3, 5}
            )
            base_energy = base_state.getPotentialEnergy().value_in_unit(
                unit.kilojoules_per_mole
            )
            bias_state = self.sampler.context.getState(
                getEnergy=True, groups={1, 4}
            )
            sampling_bias = bias_state.getPotentialEnergy().value_in_unit(
                unit.kilojoules_per_mole
            )
            original = tuple(
                float(value)
                for value in self.sampler._collect_interaction_energies()
            )
            lrc = tuple(
                float(value)
                for value in self.sampler._lj_tail_correction_kj_mol()
            )
            centered_basis = (
                self.ibs_wrapper.get_centered_basis_energies_kj_mol(
                    self.sampler.context
                )
            )
            raw_basis = tuple(
                centered + basis.energy_offset_kj_mol
                for centered, basis in zip(
                    centered_basis, self.controller.bases, strict=True
                )
            )
            frame = compose_ibs_energy_frame(
                self.controller,
                lambdas=self.lambdas,
                original_interaction_energies_kj_mol=original,
                lrc_state_energies_kj_mol=lrc,
                basis_energies_kj_mol=raw_basis,
                sampling_bias_energy_kj_mol=sampling_bias,
                base_energy_kj_mol=base_energy,
            )
            if frame.bias_cv_state_energies_kj_mol:
                self.sampler.e_offset = (
                    frame.bias_cv_state_energies_kj_mol[0]
                )
            relative_bias_cv = np.asarray(
                [
                    value - float(self.sampler.e_offset)
                    for value in frame.bias_cv_state_energies_kj_mol
                ],
                dtype=np.float64,
            )
            before = {
                "energy_buffer": len(self.sampler.energy_buffer),
                "energy_history": len(self.sampler.energy_history),
                "bias_history": len(self.sampler.bias_history),
                "base_energy_history": len(self.sampler.base_energy_history),
                "neural_path": len(self.neural_path_energy_history),
                "basis": len(self.basis_energy_history),
            }
            try:
                self.sampler.energy_buffer.append(relative_bias_cv)
                self.sampler.energy_history.append(
                    np.asarray(
                        frame.target_state_energies_kj_mol,
                        dtype=np.float64,
                    )
                )
                self.sampler.bias_history.append(
                    frame.sampling_bias_energy_kj_mol
                )
                self.sampler.base_energy_history.append(
                    frame.base_energy_kj_mol
                )
                self.neural_path_energy_history.append(
                    frame.neural_path_state_energies_kj_mol
                )
                self.basis_energy_history.append(
                    frame.basis_energies_kj_mol
                )
            except Exception:
                for name, history in (
                    ("energy_buffer", self.sampler.energy_buffer),
                    ("energy_history", self.sampler.energy_history),
                    ("bias_history", self.sampler.bias_history),
                    ("base_energy_history", self.sampler.base_energy_history),
                    ("neural_path", self.neural_path_energy_history),
                    ("basis", self.basis_energy_history),
                ):
                    del history[before[name] :]
                raise
            self._record_query(True)
            return relative_bias_cv
        except Exception as exc:
            if isinstance(exc, RuntimeError) and "hard gate" in str(exc):
                raise
            self._record_query(False, f"neural_path:{type(exc).__name__}")
            return np.full(len(self.lambdas), np.nan, dtype=np.float64)


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
    box_vectors_by_frame_nm: Sequence[Sequence[Sequence[float]]] | None = None,
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
    if box_vectors_by_frame_nm is None:
        frame_boxes = (None,) * len(frames)
    else:
        if len(box_vectors_by_frame_nm) != len(frames):
            raise NeuralPathConfigError(
                "box_vectors_by_frame_nm 数量必须与 frames 一致"
            )
        frame_boxes = tuple(
            _normalize_box_vectors_nm(box) for box in box_vectors_by_frame_nm
        )
    if (
        basis_spec.coordinate_imaging == "minimum_image_local"
        and any(box is None for box in frame_boxes)
    ):
        raise NeuralPathConfigError(
            "minimum_image_local benchmark 必须为每个 frame 提供盒向量"
        )
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
        for frame_index, (frame, frame_box) in enumerate(
            zip(frames, frame_boxes, strict=True)
        ):
            frame_started = time.perf_counter()
            evaluation_frame = frame
            if frame_box is not None:
                import numpy as np

                full_positions = np.asarray(frame, dtype=np.float64).copy()
                combined_indices = tuple(ligand_indices) + tuple(
                    environment_indices
                )
                full_positions[list(combined_indices)] = (
                    MaceDecompositionPythonComputation._minimum_image_selected(
                        full_positions,
                        list(combined_indices),
                        np.asarray(frame_box, dtype=np.float64),
                    )
                )
                evaluation_frame = full_positions.tolist()
            evaluation = adapter.evaluate(
                evaluation_frame,
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
            support = controller.evaluate_support_domains(
                frame, box_vectors_nm=frame_box
            )
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


def periodic_dihedral_degrees(
    positions_nm: Sequence[Sequence[float]],
    atom_indices: Sequence[int],
    *,
    box_vectors_nm: Sequence[Sequence[float]] | None = None,
) -> float:
    """计算四原子周期二面角，返回 ``[-180, 180)`` 度。"""

    import numpy as np

    frame = np.asarray(
        _normalize_frame_collection([positions_nm])[0], dtype=np.float64
    )
    if (
        not isinstance(atom_indices, Sequence)
        or isinstance(atom_indices, (str, bytes))
        or len(atom_indices) != 4
    ):
        raise NeuralPathConfigError("torsion atom_indices 必须恰好含 4 个整数")
    indices = []
    for position, value in enumerate(atom_indices):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= len(frame)
        ):
            raise NeuralPathConfigError(
                f"torsion atom_indices[{position}] 超出坐标范围"
            )
        indices.append(value)
    if len(set(indices)) != 4:
        raise NeuralPathConfigError("torsion atom_indices 不允许重复")
    points = frame[indices]
    box = _normalize_box_vectors_nm(box_vectors_nm)
    inverse_box = None
    box_array = None
    if box is not None:
        box_array = np.asarray(box, dtype=np.float64)
        inverse_box = np.linalg.inv(box_array)

    def displacement(left, right):
        vector = np.asarray(right - left, dtype=np.float64)
        if inverse_box is not None:
            fractional = vector @ inverse_box
            vector -= np.floor(fractional + 0.5) @ box_array
        return vector

    b0 = -displacement(points[0], points[1])
    b1 = displacement(points[1], points[2])
    b2 = displacement(points[2], points[3])
    norm_b1 = float(np.linalg.norm(b1))
    if norm_b1 <= 1.0e-12:
        raise NeuralPathFrameError("torsion 中央键长度接近零")
    axis = b1 / norm_b1
    v = b0 - np.dot(b0, axis) * axis
    w = b2 - np.dot(b2, axis) * axis
    if min(float(np.linalg.norm(v)), float(np.linalg.norm(w))) <= 1.0e-12:
        raise NeuralPathFrameError("torsion 四原子近共线，角度未定义")
    angle = math.degrees(
        math.atan2(float(np.dot(np.cross(axis, v), w)), float(np.dot(v, w)))
    )
    return ((angle + 180.0) % 360.0) - 180.0


def classify_torsion_basin(angle_degrees: float) -> str:
    """按预注册三盆定义分类 trans/gauche-/gauche+。"""

    angle = ((_finite_float(angle_degrees, "torsion angle") + 180.0) % 360.0) - 180.0
    if abs(angle) >= 120.0:
        return "trans"
    if angle < 0.0:
        return "gauche_minus"
    return "gauche_plus"


def analyze_periodic_torsion_series(
    angles_degrees: Sequence[float],
) -> dict[str, Any]:
    """汇总周期 torsion 分布，并用 core hysteresis 统计盆间转换。"""

    import numpy as np

    values = np.asarray(
        [
            ((_finite_float(value, "torsion angle") + 180.0) % 360.0)
            - 180.0
            for value in angles_degrees
        ],
        dtype=np.float64,
    )
    if values.size == 0:
        raise NeuralPathConfigError("torsion series 不能为空")
    radians = np.deg2rad(values)
    mean_vector = np.mean(np.exp(1j * radians))
    circular_mean = math.degrees(math.atan2(mean_vector.imag, mean_vector.real))
    resultant = float(abs(mean_vector))
    circular_std = (
        math.degrees(math.sqrt(max(0.0, -2.0 * math.log(resultant))))
        if resultant > 0.0
        else 180.0
    )
    basin_order = ("trans", "gauche_minus", "gauche_plus")
    basin_labels = [classify_torsion_basin(value) for value in values]
    occupancy = {
        basin: basin_labels.count(basin) / len(basin_labels)
        for basin in basin_order
    }

    def core_label(value: float) -> str | None:
        if abs(value) >= 150.0:
            return "trans"
        if -90.0 <= value <= -30.0:
            return "gauche_minus"
        if 30.0 <= value <= 90.0:
            return "gauche_plus"
        return None

    transition_counts = {
        f"{left}->{right}": 0
        for left in basin_order
        for right in basin_order
        if left != right
    }
    last_core = None
    for value in values:
        current_core = core_label(float(value))
        if current_core is None:
            continue
        if last_core is not None and current_core != last_core:
            transition_counts[f"{last_core}->{current_core}"] += 1
        last_core = current_core
    histogram, edges = np.histogram(
        values, bins=np.linspace(-180.0, 180.0, 25)
    )
    return {
        "count": int(values.size),
        "circular_mean_degrees": circular_mean,
        "circular_resultant_length": resultant,
        "circular_std_degrees": circular_std,
        "basin_definition": {
            "trans": "|phi| >= 120 deg",
            "gauche_minus": "-120 <= phi < 0 deg",
            "gauche_plus": "0 <= phi < 120 deg",
            "transition_core_hysteresis": {
                "trans": "|phi| >= 150 deg",
                "gauche_minus": "-90 <= phi <= -30 deg",
                "gauche_plus": "30 <= phi <= 90 deg",
            },
        },
        "basin_occupancy": occupancy,
        "core_transition_count": int(sum(transition_counts.values())),
        "core_transition_counts": transition_counts,
        "histogram": {
            "bin_edges_degrees": edges.tolist(),
            "counts": histogram.astype(int).tolist(),
            "probabilities": (histogram / values.size).tolist(),
        },
    }


def _atom_identity(atom: Any) -> dict[str, Any]:
    """返回跨 System 重建仍可审计的 topology 原子身份。"""

    residue = atom.residue
    chain = residue.chain
    element = getattr(atom, "element", None)
    residue_atoms = list(residue.atoms)
    return {
        "index": int(atom.index),
        "name": str(atom.name),
        "serial": (
            int(atom.serial) if getattr(atom, "serial", None) is not None else None
        ),
        "residue_atom_ordinal": residue_atoms.index(atom),
        "element": (
            str(getattr(element, "symbol", "")) if element is not None else None
        ),
        "residue_name": str(residue.name),
        "residue_id": int(getattr(residue, "resSeq", residue.index)),
        "residue_index": int(residue.index),
        "chain_index": int(chain.index),
    }


def discover_ligand_rotatable_torsions(
    topology: Any,
    ligand_indices: Sequence[int],
    *,
    bond_pairs: Sequence[Sequence[int]] | None = None,
) -> list[dict[str, Any]]:
    """从键图发现非环、非末端的 ligand 重原子中央键及确定性 torsion。"""

    ligand = {int(index) for index in ligand_indices}
    if len(ligand) < 4:
        raise NeuralPathConfigError("ligand_indices 至少需要四个原子")
    atoms = list(topology.atoms)
    if min(ligand) < 0 or max(ligand) >= len(atoms):
        raise NeuralPathConfigError("ligand_indices 超出 topology")
    neighbors = {index: set() for index in ligand}
    topology_bonds = (
        [(int(atom_a.index), int(atom_b.index)) for atom_a, atom_b in topology.bonds]
        if bond_pairs is None
        else [(int(pair[0]), int(pair[1])) for pair in bond_pairs]
    )
    for left, right in topology_bonds:
        if left in ligand and right in ligand:
            neighbors[left].add(right)
            neighbors[right].add(left)

    def is_heavy(index: int) -> bool:
        element = getattr(atoms[index], "element", None)
        return element is not None and getattr(element, "atomic_number", 0) > 1

    def central_bond_is_in_ring(left: int, right: int) -> bool:
        stack = [left]
        visited = {left}
        while stack:
            current = stack.pop()
            for neighbor in neighbors[current]:
                if {current, neighbor} == {left, right}:
                    continue
                if neighbor == right:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return False

    candidates = []
    for left in sorted(ligand):
        for right in sorted(neighbors[left]):
            if left >= right or not (is_heavy(left) and is_heavy(right)):
                continue
            outer_left = sorted(
                index
                for index in neighbors[left] - {right}
                if is_heavy(index)
            )
            outer_right = sorted(
                index
                for index in neighbors[right] - {left}
                if is_heavy(index)
            )
            if not outer_left or not outer_right:
                continue
            if central_bond_is_in_ring(left, right):
                continue

            def outer_priority(index: int):
                element = getattr(atoms[index], "element", None)
                return (
                    len(neighbors[index]),
                    int(getattr(element, "atomic_number", 0)),
                    -index,
                )

            atom_a = max(outer_left, key=outer_priority)
            atom_d = max(outer_right, key=outer_priority)
            indices = [atom_a, left, right, atom_d]
            identities = [_atom_identity(atoms[index]) for index in indices]
            candidates.append(
                {
                    "candidate_type": "ligand_rotatable_torsion",
                    "atom_indices": indices,
                    "central_bond_indices": [left, right],
                    "atoms": identities,
                    "stable_id": "ligand_torsion:"
                    + "-".join(
                        f"chain{identity['chain_index']}:"
                        f"{identity['residue_name']}:{identity['residue_id']}:"
                        f"{identity['name']}@{identity['residue_atom_ordinal']}"
                        for identity in identities
                    ),
                }
            )
    return candidates


def discover_pocket_sidechain_chi1_torsions(
    topology: Any,
    reference_positions_nm: Sequence[Sequence[float]],
    ligand_indices: Sequence[int],
    *,
    box_vectors_nm: Sequence[Sequence[float]] | None = None,
    pocket_cutoff_nm: float = 0.6,
) -> list[dict[str, Any]]:
    """按 ligand 距离发现口袋残基，并用稳定原子名定义标准 chi1。"""

    import numpy as np

    cutoff = _finite_float(pocket_cutoff_nm, "pocket_cutoff_nm")
    if cutoff <= 0.0:
        raise NeuralPathConfigError("pocket_cutoff_nm 必须为正")
    atoms = list(topology.atoms)
    positions = np.asarray(
        _normalize_frame_collection([reference_positions_nm])[0],
        dtype=np.float64,
    )
    ligand = {int(index) for index in ligand_indices}
    ligand_heavy = [
        index
        for index in ligand
        if getattr(getattr(atoms[index], "element", None), "atomic_number", 0)
        > 1
    ]
    if not ligand_heavy:
        raise NeuralPathConfigError("ligand 没有重原子")
    box = _normalize_box_vectors_nm(box_vectors_nm)
    box_array = np.asarray(box, dtype=np.float64) if box is not None else None
    inverse_box = np.linalg.inv(box_array) if box is not None else None

    def distance_to_ligand(index: int) -> float:
        delta = positions[ligand_heavy] - positions[index]
        if inverse_box is not None:
            fractional = delta @ inverse_box
            delta -= np.floor(fractional + 0.5) @ box_array
        return float(np.min(np.linalg.norm(delta, axis=1)))

    gamma_names = ("CG", "CG1", "OG", "OG1", "SG")
    water_names = {"HOH", "WAT", "SOL", "TP3", "TIP3"}
    candidates = []
    for residue in topology.residues:
        residue_atoms = list(residue.atoms)
        if residue.name.upper() in water_names:
            continue
        by_name = {atom.name.upper(): atom for atom in residue_atoms}
        gamma = next(
            (by_name[name] for name in gamma_names if name in by_name),
            None,
        )
        required = [by_name.get("N"), by_name.get("CA"), by_name.get("CB"), gamma]
        if any(atom is None for atom in required):
            continue
        if any(atom.index in ligand for atom in required):
            continue
        residue_heavy = [
            atom.index
            for atom in residue_atoms
            if getattr(getattr(atom, "element", None), "atomic_number", 0) > 1
        ]
        minimum_distance = min(
            (distance_to_ligand(index) for index in residue_heavy),
            default=math.inf,
        )
        if minimum_distance > cutoff:
            continue
        indices = [int(atom.index) for atom in required]
        candidates.append(
            {
                "candidate_type": "pocket_sidechain_chi1",
                "atom_indices": indices,
                "atoms": [_atom_identity(atom) for atom in required],
                "reference_min_ligand_distance_nm": minimum_distance,
                "stable_id": (
                    f"sidechain_chi1:chain{residue.chain.index}:"
                    f"{residue.name}:{getattr(residue, 'resSeq', residue.index)}"
                ),
            }
        )
    return candidates


def screen_periodic_torsion_candidates(
    frames_nm: Sequence[Sequence[Sequence[float]]],
    box_vectors_by_frame_nm: Sequence[Sequence[Sequence[float]]] | None,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """对候选 torsion 计算周期 IAT；只排序，不自动宣称 production CV。"""

    import numpy as np

    frames = np.asarray(frames_nm, dtype=np.float64)
    if frames.ndim != 3 or frames.shape[2] != 3:
        raise NeuralPathConfigError("frames_nm 必须是 [frames, atoms, 3]")
    if not np.all(np.isfinite(frames)):
        raise NeuralPathFrameError("frames_nm 含非有限坐标")
    if len(frames) < 10:
        raise NeuralPathConfigError("慢变量筛选至少需要 10 帧")
    if box_vectors_by_frame_nm is None:
        boxes = None
    else:
        boxes = np.asarray(box_vectors_by_frame_nm, dtype=np.float64)
        if len(boxes) != len(frames):
            raise NeuralPathConfigError("box_vectors_by_frame_nm 与 frames 不等长")
        if boxes.shape != (len(frames), 3, 3):
            raise NeuralPathConfigError(
                "box_vectors_by_frame_nm 必须是 [frames,3,3]"
            )
        if not np.all(np.isfinite(boxes)):
            raise NeuralPathFrameError("box vectors 含非有限值")

    def displacement(left, right):
        vectors = right - left
        if boxes is not None:
            inverse_boxes = np.linalg.inv(boxes)
            fractional = np.einsum(
                "fi,fij->fj", vectors, inverse_boxes
            )
            vectors = vectors - np.einsum(
                "fi,fij->fj", np.floor(fractional + 0.5), boxes
            )
        return vectors

    def torsion_series(indices):
        points = frames[:, indices, :]
        b0 = -displacement(points[:, 0], points[:, 1])
        b1 = displacement(points[:, 1], points[:, 2])
        b2 = displacement(points[:, 2], points[:, 3])
        b1_norm = np.linalg.norm(b1, axis=1)
        if np.any(b1_norm <= 1.0e-12):
            raise NeuralPathFrameError("torsion 中央键长度接近零")
        axis = b1 / b1_norm[:, None]
        v = b0 - np.sum(b0 * axis, axis=1)[:, None] * axis
        w = b2 - np.sum(b2 * axis, axis=1)[:, None] * axis
        if np.any(
            np.minimum(np.linalg.norm(v, axis=1), np.linalg.norm(w, axis=1))
            <= 1.0e-12
        ):
            raise NeuralPathFrameError("torsion 四原子近共线")
        x = np.sum(v * w, axis=1)
        y = np.sum(np.cross(axis, v) * w, axis=1)
        return np.rad2deg(np.arctan2(y, x))

    reports = []
    for candidate in candidates:
        indices = [int(index) for index in candidate["atom_indices"]]
        if len(indices) != 4 or min(indices) < 0 or max(indices) >= frames.shape[1]:
            raise NeuralPathConfigError("candidate atom_indices 超出 frame")
        angles = torsion_series(indices).tolist()
        radians = np.deg2rad(np.asarray(angles, dtype=np.float64))
        sin_iat = integrated_autocorrelation_time(np.sin(radians).tolist())
        cos_iat = integrated_autocorrelation_time(np.cos(radians).tolist())
        periodic_g = max(
            float(sin_iat["statistical_inefficiency"]),
            float(cos_iat["statistical_inefficiency"]),
        )
        summary = analyze_periodic_torsion_series(angles)
        variability = 1.0 - float(summary["circular_resultant_length"])
        report = dict(candidate)
        report.update(
            {
                "periodic_statistical_inefficiency": periodic_g,
                "effective_uncorrelated_samples": len(angles) / periodic_g,
                "variability_factor": variability,
                "screening_score": periodic_g * variability,
                "torsion": summary,
            }
        )
        reports.append(report)
    reports.sort(
        key=lambda item: (
            -float(item["screening_score"]),
            -float(item["periodic_statistical_inefficiency"]),
            str(item["stable_id"]),
        )
    )
    for rank, report in enumerate(reports, start=1):
        report["rank_within_periodic_torsions"] = rank
    return {
        "report_type": "outer_lambda_slow_variable_screen",
        "report_version": 2,
        "n_frames": len(frames),
        "selection_status": "candidate_ranking_only",
        "selection_warning": (
            "排名只表示当前困难窗口轨迹中的慢且有变化；必须结合重复轨迹、"
            "状态转换和 ESS 关联后才能冻结 production CV"
        ),
        "periodic_torsion_candidates": reports,
    }


def screen_ligand_hydration_coordination(
    frames_nm: Sequence[Sequence[Sequence[float]]],
    box_vectors_by_frame_nm: Sequence[Sequence[Sequence[float]]] | None,
    topology: Any,
    ligand_indices: Sequence[int],
    *,
    switching_distance_nm: float = 0.35,
    switching_power: int = 6,
) -> dict[str, Any]:
    """筛选 ligand 第一水合壳层；结果是候选描述符，不是已冻结的生产 CV。"""

    import numpy as np

    frames = np.asarray(frames_nm, dtype=np.float64)
    if frames.ndim != 3 or frames.shape[2] != 3 or len(frames) < 10:
        raise NeuralPathConfigError(
            "hydration screening 需要至少 10 帧 [frames, atoms, 3] 坐标"
        )
    if not np.all(np.isfinite(frames)):
        raise NeuralPathFrameError("hydration screening 坐标含非有限值")
    atoms = list(topology.atoms)
    if len(atoms) != frames.shape[1]:
        raise NeuralPathConfigError("topology 原子数与 hydration frames 不一致")
    ligand = {int(index) for index in ligand_indices}
    if not ligand or min(ligand) < 0 or max(ligand) >= len(atoms):
        raise NeuralPathConfigError("ligand_indices 超出 hydration frames")
    ligand_heavy = [
        index
        for index in sorted(ligand)
        if getattr(getattr(atoms[index], "element", None), "atomic_number", 0)
        > 1
    ]
    if not ligand_heavy:
        raise NeuralPathConfigError("ligand 没有可用于 hydration CV 的重原子")
    water_names = {"HOH", "WAT", "SOL", "TP3", "TIP3", "TIP3P"}
    water_oxygens = [
        int(atom.index)
        for atom in atoms
        if atom.residue.name.upper() in water_names
        and getattr(getattr(atom, "element", None), "atomic_number", 0) == 8
    ]
    if not water_oxygens:
        raise NeuralPathConfigError("topology 中没有识别到水氧原子")
    cutoff = _finite_float(
        switching_distance_nm, "hydration.switching_distance_nm"
    )
    if cutoff <= 0.0:
        raise NeuralPathConfigError(
            "hydration.switching_distance_nm 必须为正"
        )
    if (
        isinstance(switching_power, bool)
        or not isinstance(switching_power, int)
        or switching_power <= 0
    ):
        raise NeuralPathConfigError("hydration.switching_power 必须为正整数")
    boxes = None
    inverse_boxes = None
    if box_vectors_by_frame_nm is not None:
        boxes = np.asarray(box_vectors_by_frame_nm, dtype=np.float64)
        if boxes.shape != (len(frames), 3, 3):
            raise NeuralPathConfigError(
                "hydration box vectors 必须是 [frames,3,3]"
            )
        if not np.all(np.isfinite(boxes)):
            raise NeuralPathFrameError("hydration box vectors 含非有限值")
        inverse_boxes = np.linalg.inv(boxes)

    values = []
    for frame_index, frame in enumerate(frames):
        displacement = (
            frame[np.asarray(water_oxygens), None, :]
            - frame[None, np.asarray(ligand_heavy), :]
        )
        if boxes is not None:
            fractional = displacement @ inverse_boxes[frame_index]
            displacement -= (
                np.floor(fractional + 0.5) @ boxes[frame_index]
            )
        minimum_distance = np.min(
            np.linalg.norm(displacement, axis=2), axis=1
        )
        coordination = np.sum(
            1.0 / (1.0 + (minimum_distance / cutoff) ** switching_power)
        )
        values.append(float(coordination))

    array = np.asarray(values, dtype=np.float64)
    iat = integrated_autocorrelation_time(values)
    mean = float(np.mean(array))
    std = float(np.std(array))
    percentiles = np.percentile(array, [5.0, 25.0, 50.0, 75.0, 95.0])
    return {
        "candidate_type": "ligand_first_shell_hydration",
        "stable_id": (
            f"ligand_hydration:min_heavy_distance:r0={cutoff:.6g}:"
            f"power={switching_power}"
        ),
        "definition": {
            "water_site": "water oxygen",
            "ligand_site": "nearest ligand heavy atom",
            "switching_function": "sum_w 1/(1+(min_i(r_wi)/r0)^power)",
            "switching_distance_nm": cutoff,
            "switching_power": switching_power,
            "periodic_minimum_image": boxes is not None,
        },
        "ligand_heavy_atom_count": len(ligand_heavy),
        "water_oxygen_count": len(water_oxygens),
        "count": len(values),
        "mean": mean,
        "std": std,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "percentiles": {
            "p05": float(percentiles[0]),
            "p25": float(percentiles[1]),
            "p50": float(percentiles[2]),
            "p75": float(percentiles[3]),
            "p95": float(percentiles[4]),
        },
        "relative_std": std / max(abs(mean), 1.0),
        "statistical_inefficiency": float(
            iat["statistical_inefficiency"]
        ),
        "effective_uncorrelated_samples": float(
            iat["effective_uncorrelated_samples"]
        ),
        "selection_status": "candidate_ranking_only",
    }


def compare_slow_variable_screens(
    screen_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """按 stable_id 对齐独立种子；至少三条轨迹才允许自动给出资格结论。"""

    if len(screen_reports) < 2:
        raise NeuralPathConfigError("慢变量重复比较至少需要两个 screen report")
    normalized = []
    for report_index, report in enumerate(screen_reports):
        if not isinstance(report, Mapping):
            raise NeuralPathConfigError(
                f"screen_reports[{report_index}] 必须是映射"
            )
        if report.get("report_type") != "outer_lambda_slow_variable_screen":
            raise NeuralPathConfigError(
                f"screen_reports[{report_index}] report_type 不正确"
            )
        candidates = report.get("periodic_torsion_candidates")
        if not isinstance(candidates, list):
            raise NeuralPathConfigError(
                f"screen_reports[{report_index}] 缺少 torsion candidates"
            )
        by_id = {}
        for candidate in candidates:
            stable_id = _nonempty_string(
                candidate.get("stable_id"), "candidate.stable_id"
            )
            if stable_id in by_id:
                raise NeuralPathConfigError(
                    f"同一 screen report 中 stable_id 重复: {stable_id}"
                )
            by_id[stable_id] = candidate
        normalized.append((report, by_id))

    common_ids = set(normalized[0][1])
    for _, by_id in normalized[1:]:
        common_ids &= set(by_id)
    required_transition_runs = math.ceil(len(normalized) / 2)
    periodic = []
    for stable_id in sorted(common_ids):
        observations = [by_id[stable_id] for _, by_id in normalized]
        transition_counts = [
            int(item["torsion"]["core_transition_count"])
            for item in observations
        ]
        transition_positive_runs = sum(
            count > 0 for count in transition_counts
        )
        median_g = statistics.median(
            float(item["periodic_statistical_inefficiency"])
            for item in observations
        )
        median_std = statistics.median(
            float(item["torsion"]["circular_std_degrees"])
            for item in observations
        )
        reproducible_switching = (
            transition_positive_runs >= required_transition_runs
            and median_g >= 5.0
            and median_std >= 15.0
        )
        periodic.append(
            {
                "stable_id": stable_id,
                "candidate_type": observations[0].get("candidate_type"),
                "atom_indices": observations[0].get("atom_indices"),
                "atoms": observations[0].get("atoms"),
                "runs_present": len(observations),
                "rank_by_run": [
                    int(item["rank_within_periodic_torsions"])
                    for item in observations
                ],
                "statistical_inefficiency_by_run": [
                    float(item["periodic_statistical_inefficiency"])
                    for item in observations
                ],
                "core_transition_count_by_run": transition_counts,
                "transition_positive_runs": transition_positive_runs,
                "median_statistical_inefficiency": median_g,
                "median_circular_std_degrees": median_std,
                "classification": (
                    "reproducible_switching_candidate"
                    if reproducible_switching
                    else "not_yet_reproducible"
                ),
            }
        )
    periodic.sort(
        key=lambda item: (
            item["classification"] != "reproducible_switching_candidate",
            -int(item["transition_positive_runs"]),
            -float(item["median_statistical_inefficiency"]),
            str(item["stable_id"]),
        )
    )
    for rank, candidate in enumerate(periodic, start=1):
        candidate["replicate_rank"] = rank

    hydration_observations = []
    hydration_id = None
    for report, _ in normalized:
        hydration = report.get("hydration_candidate")
        if not isinstance(hydration, Mapping):
            hydration_observations = []
            break
        current_id = hydration.get("stable_id")
        if hydration_id is None:
            hydration_id = current_id
        if current_id != hydration_id:
            raise NeuralPathConfigError(
                "hydration candidate 定义在重复之间不一致"
            )
        hydration_observations.append(hydration)
    hydration_summary = None
    if hydration_observations:
        hydration_summary = {
            "stable_id": hydration_id,
            "statistical_inefficiency_by_run": [
                float(item["statistical_inefficiency"])
                for item in hydration_observations
            ],
            "relative_std_by_run": [
                float(item["relative_std"])
                for item in hydration_observations
            ],
            "median_statistical_inefficiency": statistics.median(
                float(item["statistical_inefficiency"])
                for item in hydration_observations
            ),
            "median_relative_std": statistics.median(
                float(item["relative_std"])
                for item in hydration_observations
            ),
            "classification": "descriptor_only_requires_state_definition",
        }

    three_or_more = len(normalized) >= 3
    qualified = [
        item["stable_id"]
        for item in periodic
        if item["classification"] == "reproducible_switching_candidate"
    ]
    return {
        "report_type": "outer_lambda_slow_variable_replicate_comparison",
        "report_version": 1,
        "n_reports": len(normalized),
        "common_periodic_candidate_count": len(periodic),
        "minimum_reports_for_freeze": 3,
        "selection_status": (
            "qualified_candidates_available"
            if three_or_more and qualified
            else "more_independent_sampling_required"
            if not three_or_more
            else "no_reproducible_switching_candidate"
        ),
        "production_cv_may_be_frozen": bool(three_or_more and qualified),
        "qualified_periodic_stable_ids": qualified if three_or_more else [],
        "periodic_candidates": periodic,
        "hydration_candidate": hydration_summary,
        "warning": (
            "该门只验证跨种子动力学可重复性；冻结 production CV 前仍需确认"
            "其与困难 window ESS/状态混合相关，且 hydration 需要预定义 wet/dry 状态"
        ),
    }


def freeze_slow_variable_manifest(comparison_report: Mapping[str, Any], difficult_window: Mapping[str, Any], *, replicate_rank: int=1, experiment_id: str='EXP-010') -> dict[str, Any]:
    """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""

    from archive.outer_lambda_exp010_exp011_legacy import (
        freeze_slow_variable_manifest as _legacy_implementation,
    )

    return _legacy_implementation(comparison_report, difficult_window, replicate_rank=replicate_rank, experiment_id=experiment_id)


def torsion_coordinate_gradient_radians(positions_nm: Sequence[Sequence[float]], atom_indices: Sequence[int], *, box_vectors_nm: Sequence[Sequence[float]] | None=None, displacement_nm: float=1e-06) -> tuple[tuple[float, float, float], ...]:
    """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""

    from archive.outer_lambda_exp010_exp011_legacy import (
        torsion_coordinate_gradient_radians as _legacy_implementation,
    )

    return _legacy_implementation(positions_nm, atom_indices, box_vectors_nm=box_vectors_nm, displacement_nm=displacement_nm)


def build_exp010_protein_only_selection(selection_meta: Mapping[str, Any], topology: Any) -> dict[str, Any]:
    """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""

    from archive.outer_lambda_exp010_exp011_legacy import (
        build_exp010_protein_only_selection as _legacy_implementation,
    )

    return _legacy_implementation(selection_meta, topology)


def project_force_onto_torsion(forces_kj_mol_nm: Sequence[Sequence[float]], torsion_gradient_radian_per_nm: Sequence[Sequence[float]]) -> dict[str, float]:
    """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""

    from archive.outer_lambda_exp010_exp011_legacy import (
        project_force_onto_torsion as _legacy_implementation,
    )

    return _legacy_implementation(forces_kj_mol_nm, torsion_gradient_radian_per_nm)


def build_exp010_teacher_dataset(adapter: Any, frame_records: Iterable[Mapping[str, Any]], slow_variable_manifest: Mapping[str, Any], *, ligand_indices: Sequence[int], environment_indices: Sequence[int], atomic_numbers: Sequence[int], energy_offset_kj_mol: float | None, include_secondary: bool=True) -> dict[str, Any]:
    """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""

    from archive.outer_lambda_exp010_exp011_legacy import (
        build_exp010_teacher_dataset as _legacy_implementation,
    )

    return _legacy_implementation(adapter, frame_records, slow_variable_manifest, ligand_indices=ligand_indices, environment_indices=environment_indices, atomic_numbers=atomic_numbers, energy_offset_kj_mol=energy_offset_kj_mol, include_secondary=include_secondary)


def _periodic_fourier_wavevectors(dimensions: int, order: int) -> list[tuple[int, ...]]:
    """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""

    from archive.outer_lambda_exp010_exp011_legacy import (
        _periodic_fourier_wavevectors as _legacy_implementation,
    )

    return _legacy_implementation(dimensions, order)


def _fourier_design_matrix(angles, wavevectors):
    import numpy as np

    phase = np.asarray(angles, dtype=np.float64) @ np.asarray(
        wavevectors, dtype=np.float64
    ).T
    return np.column_stack(
        [np.ones(len(phase), dtype=np.float64), np.cos(phase), np.sin(phase)]
    )


def _regression_metrics(observed, predicted) -> dict[str, float]:
    import numpy as np

    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    residual = predicted - observed
    rmse = float(np.sqrt(np.mean(residual * residual)))
    mae = float(np.mean(np.abs(residual)))
    variance = float(np.sum((observed - np.mean(observed)) ** 2))
    r2 = (
        1.0 - float(np.sum(residual * residual)) / variance
        if variance > 0.0
        else 0.0
    )
    correlation = (
        float(np.corrcoef(observed, predicted)[0, 1])
        if len(observed) >= 2
        and float(np.std(observed)) > 0.0
        and float(np.std(predicted)) > 0.0
        else 0.0
    )
    return {
        "count": int(len(observed)),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_correlation": correlation,
    }


def fit_periodic_fourier_distillation(teacher_dataset: Mapping[str, Any], *, dimensions: int=1, order: int=4, ridge: float=1e-06, conditional_bins: int=24) -> dict[str, Any]:
    """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""

    from archive.outer_lambda_exp010_exp011_legacy import (
        fit_periodic_fourier_distillation as _legacy_implementation,
    )

    return _legacy_implementation(teacher_dataset, dimensions=dimensions, order=order, ridge=ridge, conditional_bins=conditional_bins)


def build_periodic_fourier_openmm_force(model: Mapping[str, Any], *, force_group: int=0) -> Any:
    """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""

    from archive.outer_lambda_exp010_exp011_legacy import (
        build_periodic_fourier_openmm_force as _legacy_implementation,
    )

    return _legacy_implementation(model, force_group=force_group)


def build_exp011_periodic_umbrella_force(atom_indices: Sequence[int], *, center_degrees: float, force_constant_kj_mol_radian2: float, force_group: int=31):
    """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""

    from archive.outer_lambda_exp010_exp011_legacy import (
        build_exp011_periodic_umbrella_force as _legacy_implementation,
    )

    return _legacy_implementation(atom_indices, center_degrees=center_degrees, force_constant_kj_mol_radian2=force_constant_kj_mol_radian2, force_group=force_group)


def run_hard_window_scratch_trajectory(baseline_root: str | Path, output_dir: str | Path, *, window_index: int=0, initial_trajectory_path: str | Path | None=None, burnin_steps: int=10000, sampling_steps: int=100000, report_interval_steps: int=500, platform_name: str='CUDA', random_seed: int=20260731, umbrella_torsion_atom_indices: Sequence[int] | None=None, umbrella_center_degrees: float | None=None, umbrella_force_constant_kj_mol_radian2: float | None=None, umbrella_run_id: str | None=None, minimize_max_iterations: int=200) -> dict[str, Any]:
    """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""

    from archive.outer_lambda_exp010_exp011_legacy import (
        run_hard_window_scratch_trajectory as _legacy_implementation,
    )

    return _legacy_implementation(baseline_root, output_dir, window_index=window_index, initial_trajectory_path=initial_trajectory_path, burnin_steps=burnin_steps, sampling_steps=sampling_steps, report_interval_steps=report_interval_steps, platform_name=platform_name, random_seed=random_seed, umbrella_torsion_atom_indices=umbrella_torsion_atom_indices, umbrella_center_degrees=umbrella_center_degrees, umbrella_force_constant_kj_mol_radian2=umbrella_force_constant_kj_mol_radian2, umbrella_run_id=umbrella_run_id, minimize_max_iterations=minimize_max_iterations)


def select_wp0_difficult_window(
    final_results: Mapping[str, Any],
) -> dict[str, Any]:
    """从已完成 Stage-2 诊断中按最低最终 ESS ratio 选择困难窗口。"""

    try:
        windows = final_results["stage_diagnostics"]["stage2"][
            "window_overlap_diagnostics"
        ]
    except (KeyError, TypeError) as exc:
        raise NeuralPathConfigError(
            "final results 缺少 stage2.window_overlap_diagnostics"
        ) from exc
    if (
        not isinstance(windows, Sequence)
        or isinstance(windows, (str, bytes))
        or not windows
    ):
        raise NeuralPathConfigError("window_overlap_diagnostics 必须是非空序列")
    normalized = []
    for position, window in enumerate(windows):
        if not isinstance(window, Mapping):
            raise NeuralPathConfigError(f"window[{position}] 必须是映射")
        normalized.append(
            {
                "window_index": int(window["window_index"]),
                "window_range": list(window.get("window_range", [])),
                "lambdas_vdw": list(window.get("lambdas_vdw", [])),
                "min_ess_ratio": _finite_float(
                    window.get("min_ess_ratio"),
                    f"window[{position}].min_ess_ratio",
                ),
                "absolute_ess": _finite_float(
                    window.get("absolute_ess"),
                    f"window[{position}].absolute_ess",
                ),
                "statistical_inefficiency": _finite_float(
                    window.get("statistical_inefficiency"),
                    f"window[{position}].statistical_inefficiency",
                ),
                "endpoint_diff_uncertainty_kj_mol": _finite_float(
                    window.get("endpoint_diff_uncertainty_kJ_mol"),
                    f"window[{position}].endpoint uncertainty",
                ),
            }
        )
    ranked = sorted(
        normalized,
        key=lambda item: (
            item["min_ess_ratio"],
            item["absolute_ess"],
            -item["statistical_inefficiency"],
        ),
    )
    return {
        "selection_rule": (
            "minimum final stage2 min_ess_ratio; ties use lower absolute_ess "
            "then higher statistical_inefficiency"
        ),
        "selected_window": ranked[0],
        "ranked_windows": ranked,
    }


def build_wp0_selection_report(
    *,
    final_results_path: str | Path,
    topology_path: str | Path,
    trajectory_paths: Sequence[str | Path],
    torsion_atom_indices: Sequence[int],
    slow_variable_name: str,
) -> dict[str, Any]:
    """生成冻结困难窗口与首个 torsion 慢变量的 WP-0 报告。"""

    final_results = _cli_read_json_mapping(final_results_path, "final-results")
    topology = Path(topology_path).expanduser().resolve()
    if not topology.is_file():
        raise NeuralPathIntegrityError(f"topology 文件不存在: {topology}")
    if not trajectory_paths:
        raise NeuralPathConfigError("至少需要一条基线 trajectory")
    try:
        import mdtraj as md
    except ImportError as exc:
        raise NeuralPathConfigError("WP-0 trajectory 分析需要 mdtraj") from exc
    trajectory_reports = []
    for raw_path in trajectory_paths:
        trajectory = Path(raw_path).expanduser().resolve()
        if not trajectory.is_file():
            raise NeuralPathIntegrityError(
                f"trajectory 文件不存在: {trajectory}"
            )
        angles = []
        for chunk in md.iterload(
            str(trajectory), top=str(topology), chunk=500
        ):
            values = md.compute_dihedrals(
                chunk,
                [list(torsion_atom_indices)],
                periodic=True,
            )
            angles.extend(
                math.degrees(float(value)) for value in values[:, 0]
            )
        trajectory_reports.append(
            {
                "trajectory_path": str(trajectory),
                "trajectory_sha256": sha256_file(trajectory),
                "torsion": analyze_periodic_torsion_series(angles),
            }
        )
    final_path = Path(final_results_path).expanduser().resolve()
    return {
        "report_type": "outer_lambda_wp0_selection",
        "report_version": 1,
        "hard_window": select_wp0_difficult_window(final_results),
        "slow_variable": {
            "name": _nonempty_string(
                slow_variable_name, "slow_variable_name"
            ),
            "type": "periodic_torsion",
            "atom_indices": [int(value) for value in torsion_atom_indices],
            "period_degrees": 360.0,
            "range_degrees": [-180.0, 180.0],
        },
        "trajectory_reports": trajectory_reports,
        "baseline_coordinate_gap": (
            "completed IBS window files contain energy/bias/base histories but "
            "no window coordinate trajectory; EXP-009 N=1 must record the "
            "selected torsion before WP-0 transition counts are final"
        ),
        "final_results_path": str(final_path),
        "final_results_sha256": sha256_file(final_path),
        "topology_path": str(topology),
        "topology_sha256": sha256_file(topology),
    }


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
    min_pair_distance_nm: float | None = None,
    max_pair_distance_nm: float | None = None,
    max_radius_of_gyration_nm: float | None = None,
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
    support_values = {
        "min_pair_distance_nm": min_pair_distance_nm,
        "max_pair_distance_nm": max_pair_distance_nm,
        "max_radius_of_gyration_nm": max_radius_of_gyration_nm,
    }
    support_payload = {
        name: _finite_float(value, name)
        for name, value in support_values.items()
        if value is not None
    }
    if support_payload:
        # 在写配置前执行与加载器相同的完整关系校验。
        NeuralBasisSupportDomain.from_mapping(support_payload)

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    indices_path = destination / "fixed_local_atom_indices.json"
    config_path = destination / "outer_lambda_existing_model.json"
    indices_payload = sorted(combined)
    indices_path.write_text(
        json.dumps(indices_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    basis_payload = {
        "name": "existing_local_interaction_basis",
        "backend": "existing_openmmml",
        "model_name": _nonempty_string(model_name, "model_name"),
        "model_path": str(artifact_path),
        "sha256": artifact_sha,
        "energy_offset_kj_mol": offset,
        "atom_selection": "fixed_indices",
        "atom_indices_path": str(indices_path),
        "output_unit": "kJ_per_mol",
        "precision": "single",
        "periodic": False,
        "coordinate_imaging": "minimum_image_local",
    }
    if support_payload:
        basis_payload["support_domain"] = support_payload
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
            "bases": [basis_payload],
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


def _cli_four_atom_indices(value: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "atom indices 必须是逗号分隔整数"
        ) from exc
    if len(values) != 4 or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(
            "atom indices 必须恰好是四个非负整数"
        )
    if len(set(values)) != 4:
        raise argparse.ArgumentTypeError("atom indices 不允许重复")
    return values


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

    def add_mace_nvt_arguments(
        command_parser: argparse.ArgumentParser,
        *,
        default_steps: int,
        default_report_interval: int,
    ) -> None:
        add_common_config(command_parser)
        command_parser.add_argument("--system-xml", required=True)
        command_parser.add_argument("--trajectory", required=True)
        command_parser.add_argument("--topology", required=True)
        command_parser.add_argument("--selection-meta", required=True)
        command_parser.add_argument("--frame", default="last")
        command_parser.add_argument(
            "--lambda", dest="lambda_value", type=float, default=0.5
        )
        command_parser.add_argument("--steps", type=int, default=default_steps)
        command_parser.add_argument(
            "--report-interval", type=int, default=default_report_interval
        )
        command_parser.add_argument(
            "--timestep-fs", type=float, default=0.5
        )
        command_parser.add_argument(
            "--temperature-k", type=float, default=300.0
        )
        command_parser.add_argument(
            "--friction-per-ps", type=float, default=1.0
        )
        command_parser.add_argument(
            "--device", choices=("cpu", "cuda"), default="cuda"
        )
        command_parser.add_argument("--platform", default="CUDA")
        command_parser.add_argument("--seed", type=int, default=20260730)

    mace_nvt_parser = subparsers.add_parser(
        "mace-nvt-smoke",
        help="快速检查每步 MACE 分解 Force 的 OpenMM 连通性",
    )
    add_mace_nvt_arguments(
        mace_nvt_parser, default_steps=10, default_report_interval=1
    )

    mace_qualification_parser = subparsers.add_parser(
        "mace-nvt-qualification",
        help="运行冻结协议的 WP-4 MACE NVT 资格测试并执行硬门判定",
    )
    add_mace_nvt_arguments(
        mace_qualification_parser,
        default_steps=1000,
        default_report_interval=25,
    )
    mace_qualification_parser.add_argument(
        "--minimum-steps", type=int, default=1000
    )
    mace_qualification_parser.add_argument(
        "--max-path-force-kj-mol-nm", type=float, default=250.0
    )
    mace_qualification_parser.add_argument(
        "--max-energy-closure-error-kj-mol", type=float, default=0.1
    )
    mace_qualification_parser.add_argument(
        "--max-integration-seconds-per-step", type=float, default=0.2
    )

    mace_mts_parser = subparsers.add_parser(
        "mace-mts-qualification",
        help="运行冻结 coefficient=0.09 的 EXP-009 N=1/2/4 MTS 矩阵",
    )
    add_common_config(mace_mts_parser)
    mace_mts_parser.add_argument("--system-xml", required=True)
    mace_mts_parser.add_argument("--trajectory", required=True)
    mace_mts_parser.add_argument("--topology", required=True)
    mace_mts_parser.add_argument("--selection-meta", required=True)
    mace_mts_parser.add_argument("--frame", default="last")
    mace_mts_parser.add_argument(
        "--torsion-indices",
        required=True,
        type=_cli_four_atom_indices,
    )
    mace_mts_parser.add_argument(
        "--lambda", dest="lambda_value", type=float, default=0.5
    )
    mace_mts_parser.add_argument("--inner-steps", type=int, default=10_000)
    mace_mts_parser.add_argument(
        "--report-interval-inner-steps", type=int, default=100
    )
    mace_mts_parser.add_argument(
        "--inner-timestep-fs", type=float, default=0.5
    )
    mace_mts_parser.add_argument(
        "--temperature-k", type=float, default=300.0
    )
    mace_mts_parser.add_argument(
        "--friction-per-ps", type=float, default=1.0
    )
    mace_mts_parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda"
    )
    mace_mts_parser.add_argument("--platform", default="CUDA")
    mace_mts_parser.add_argument("--seed", type=int, default=20260730)
    mace_mts_parser.add_argument(
        "--minimum-n4-ns-per-day", type=float, required=True
    )

    wp0_parser = subparsers.add_parser(
        "wp0-select",
        help="从现有 Stage-2 结果冻结困难窗口并分析首个周期 torsion",
    )
    wp0_parser.add_argument("--final-results", required=True)
    wp0_parser.add_argument("--topology", required=True)
    wp0_parser.add_argument(
        "--trajectory",
        required=True,
        action="append",
        help="可重复提供多条基线轨迹",
    )
    wp0_parser.add_argument(
        "--torsion-indices",
        required=True,
        type=_cli_four_atom_indices,
        help="四个逗号分隔的 system atom indices",
    )
    wp0_parser.add_argument(
        "--slow-variable-name",
        default="atenolol_C4_N2_C9_C10",
    )
    wp0_parser.add_argument("-o", "--output")

    slow_screen_parser = subparsers.add_parser(
        "screen-slow-variables",
        help="从坐标轨迹自动发现并排序 ligand torsion 与口袋残基 chi1",
    )
    slow_screen_parser.add_argument("--trajectory", required=True)
    slow_screen_parser.add_argument("--topology", required=True)
    slow_screen_parser.add_argument(
        "--ligand-indices",
        required=True,
        help="含 ligand_indices 数组的 JSON 文件",
    )
    slow_screen_parser.add_argument(
        "--system-xml",
        help="可选；从 HarmonicBondForce 读取 ligand 键图，补足 CIF 缺失键",
    )
    slow_screen_parser.add_argument(
        "--frames",
        default="all",
        help="all、last、tail:N、start:stop:step 或逗号分隔索引",
    )
    slow_screen_parser.add_argument(
        "--pocket-cutoff-nm", type=float, default=0.6
    )
    slow_screen_parser.add_argument(
        "--hydration-switching-distance-nm", type=float, default=0.35
    )
    slow_screen_parser.add_argument(
        "--hydration-switching-power", type=int, default=6
    )
    slow_screen_parser.add_argument("-o", "--output")

    slow_compare_parser = subparsers.add_parser(
        "compare-slow-variable-screens",
        help="按 stable_id 比较独立随机种子的慢变量筛选报告",
    )
    slow_compare_parser.add_argument(
        "--input",
        required=True,
        action="append",
        help="可重复提供 candidate_screen.json；至少两个",
    )
    slow_compare_parser.add_argument("-o", "--output")

    slow_freeze_parser = subparsers.add_parser(
        "freeze-slow-variable",
        help="把已通过三种子门的周期候选冻结为指定实验输入 manifest",
    )
    slow_freeze_parser.add_argument("--comparison", required=True)
    slow_freeze_parser.add_argument("--final-results", required=True)
    slow_freeze_parser.add_argument("--replicate-rank", type=int, default=1)
    slow_freeze_parser.add_argument(
        "--experiment",
        choices=("exp010", "exp011"),
        default="exp010",
    )
    slow_freeze_parser.add_argument("-o", "--output")

    exp011_coverage_parser = subparsers.add_parser(
        "exp011-coverage",
        help="按冻结协议诊断三条 run 的周期覆盖、重叠和有效样本数",
    )
    exp011_coverage_parser.add_argument("--protocol", required=True)
    exp011_coverage_parser.add_argument(
        "--trajectory", action="append"
    )
    exp011_coverage_parser.add_argument(
        "--screen-report",
        action="append",
        help="可重复提供已有 candidate_screen JSON，避免重新读取 DCD",
    )
    exp011_coverage_parser.add_argument("--topology", required=True)
    exp011_coverage_parser.add_argument("--manifest", required=True)
    exp011_coverage_parser.add_argument(
        "--frames", default="all", help="all 或每条轨迹共用的 frame spec"
    )
    exp011_coverage_parser.add_argument("-o", "--output")

    exp011_fit_parser = subparsers.add_parser(
        "exp011-fit-pmf",
        help="从显式目标权重样本拟合周期 PMF，并执行整条 run 留一硬门",
    )
    exp011_fit_parser.add_argument("--protocol", required=True)
    exp011_fit_parser.add_argument("--dataset", required=True)
    exp011_fit_parser.add_argument("-o", "--output")

    exp011_umbrella_parser = subparsers.add_parser(
        "exp011-umbrella-sample",
        help="在完整困难窗口 MM System 上运行单个周期 torsion umbrella window",
    )
    exp011_umbrella_parser.add_argument("--baseline-root", required=True)
    exp011_umbrella_parser.add_argument("--manifest", required=True)
    exp011_umbrella_parser.add_argument("--protocol", required=True)
    exp011_umbrella_parser.add_argument("--output-dir", required=True)
    exp011_umbrella_parser.add_argument("--run-id", required=True)
    exp011_umbrella_parser.add_argument("--center-degrees", required=True, type=float)
    exp011_umbrella_parser.add_argument(
        "--force-constant-kj-mol-radian2", type=float, default=100.0
    )
    exp011_umbrella_parser.add_argument("--initial-trajectory")
    exp011_umbrella_parser.add_argument("--burnin-steps", type=int, default=1000)
    exp011_umbrella_parser.add_argument(
        "--minimize-max-iterations", type=int, default=200
    )
    exp011_umbrella_parser.add_argument("--sampling-steps", type=int, default=5000)
    exp011_umbrella_parser.add_argument("--report-interval-steps", type=int, default=500)
    exp011_umbrella_parser.add_argument("--platform", default="Reference")
    exp011_umbrella_parser.add_argument("--seed", type=int, default=20260802)
    exp011_umbrella_parser.add_argument("-o", "--output")

    exp011_reweight_parser = subparsers.add_parser(
        "exp011-reweight-umbrella",
        help="对多个 umbrella window 去相关并用 MBAR 导出目标权重",
    )
    exp011_reweight_parser.add_argument("--protocol", required=True)
    exp011_reweight_parser.add_argument(
        "--input", required=True, action="append", help="可重复提供 umbrella report JSON"
    )
    exp011_reweight_parser.add_argument("--minimum-neighbor-overlap", type=float, default=0.03)
    exp011_reweight_parser.add_argument("--output-dataset", required=True)
    exp011_reweight_parser.add_argument("-o", "--output")

    exp010_label_parser = subparsers.add_parser(
        "exp010-label",
        help="用冻结 MACE 为慢变量轨迹生成能量和广义力教师数据集",
    )
    exp010_label_parser.add_argument("-c", "--config", required=True)
    exp010_label_parser.add_argument("--manifest", required=True)
    exp010_label_parser.add_argument(
        "--trajectory", required=True, action="append"
    )
    exp010_label_parser.add_argument("--topology", required=True)
    exp010_label_parser.add_argument("--selection-meta", required=True)
    exp010_label_parser.add_argument(
        "--frames",
        default="::5",
        help="每条轨迹独立应用的 frame spec；默认每 5 帧",
    )
    exp010_label_parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda"
    )
    exp010_label_parser.add_argument(
        "--primary-only", action="store_true"
    )
    exp010_label_parser.add_argument(
        "--energy-offset-mode",
        choices=("dataset_mean", "config"),
        default="dataset_mean",
    )
    exp010_label_parser.add_argument(
        "--support-violation-policy",
        choices=("exclude", "reject"),
        default="exclude",
    )
    exp010_label_parser.add_argument(
        "--max-support-exclusion-fraction", type=float, default=0.05
    )
    exp010_label_parser.add_argument("-o", "--output")

    exp010_selection_parser = subparsers.add_parser(
        "exp010-prepare-selection",
        help="从旧局部选择移除交换水，冻结 protein-only 教师环境",
    )
    exp010_selection_parser.add_argument("--selection-meta", required=True)
    exp010_selection_parser.add_argument("--topology", required=True)
    exp010_selection_parser.add_argument(
        "--output-selection-meta", required=True
    )
    exp010_selection_parser.add_argument("-o", "--output")

    exp010_fit_parser = subparsers.add_parser(
        "exp010-fit",
        help="拟合周期 1D/2D Fourier cheap-CV 并做整条 run 留一验证",
    )
    exp010_fit_parser.add_argument("--dataset", required=True)
    exp010_fit_parser.add_argument(
        "--dimensions", type=int, choices=(1, 2), default=1
    )
    exp010_fit_parser.add_argument("--order", type=int, default=4)
    exp010_fit_parser.add_argument("--ridge", type=float, default=1.0e-6)
    exp010_fit_parser.add_argument(
        "--conditional-bins", type=int, default=24
    )
    exp010_fit_parser.add_argument("-o", "--output")

    hard_window_parser = subparsers.add_parser(
        "sample-hard-window-scratch",
        help="只读历史 IBS 协议，在独立目录生成困难窗口 CV-screening 轨迹",
    )
    hard_window_parser.add_argument("--baseline-root", required=True)
    hard_window_parser.add_argument("--output-dir", required=True)
    hard_window_parser.add_argument("--window-index", type=int, default=0)
    hard_window_parser.add_argument("--initial-trajectory")
    hard_window_parser.add_argument("--burnin-steps", type=int, default=10_000)
    hard_window_parser.add_argument(
        "--sampling-steps", type=int, default=100_000
    )
    hard_window_parser.add_argument(
        "--report-interval-steps", type=int, default=500
    )
    hard_window_parser.add_argument("--platform", default="CUDA")
    hard_window_parser.add_argument("--seed", type=int, default=20260731)
    hard_window_parser.add_argument("-o", "--output")

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
    prepare_parser.add_argument("--min-pair-distance-nm", type=float)
    prepare_parser.add_argument("--max-pair-distance-nm", type=float)
    prepare_parser.add_argument("--max-radius-of-gyration-nm", type=float)
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







































































































































































































































































































































































































































































def assess_exp011_periodic_coverage(run_angles_degrees: Mapping[str, Sequence[float]], *, run_log_target_weights: Mapping[str, Sequence[float]] | None=None, run_statistical_inefficiencies: Mapping[str, float] | None=None, bins: int=24, minimum_runs: int=3, minimum_frames_per_run: int=500, minimum_effective_samples_per_run: float=25.0, minimum_occupied_fraction_per_run: float=0.5, minimum_effective_samples_per_pooled_bin: float=2.0, minimum_runs_per_bin: int=2, minimum_raw_samples_per_run_bin: int=3, minimum_pairwise_bhattacharyya: float=0.5, minimum_effective_samples_per_basin: float=5.0) -> dict[str, Any]:
    """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""

    from archive.outer_lambda_exp010_exp011_legacy import (
        assess_exp011_periodic_coverage as _legacy_implementation,
    )

    return _legacy_implementation(run_angles_degrees, run_log_target_weights=run_log_target_weights, run_statistical_inefficiencies=run_statistical_inefficiencies, bins=bins, minimum_runs=minimum_runs, minimum_frames_per_run=minimum_frames_per_run, minimum_effective_samples_per_run=minimum_effective_samples_per_run, minimum_occupied_fraction_per_run=minimum_occupied_fraction_per_run, minimum_effective_samples_per_pooled_bin=minimum_effective_samples_per_pooled_bin, minimum_runs_per_bin=minimum_runs_per_bin, minimum_raw_samples_per_run_bin=minimum_raw_samples_per_run_bin, minimum_pairwise_bhattacharyya=minimum_pairwise_bhattacharyya, minimum_effective_samples_per_basin=minimum_effective_samples_per_basin)


def fit_exp011_reweighted_periodic_pmf(dataset: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""

    from archive.outer_lambda_exp010_exp011_legacy import (
        fit_exp011_reweighted_periodic_pmf as _legacy_implementation,
    )

    return _legacy_implementation(dataset, protocol)


def reweight_exp011_umbrella_reports(reports: Sequence[Mapping[str, Any]], *, target_hamiltonian_id: str, minimum_neighbor_overlap: float=0.03) -> dict[str, Any]:
    """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""

    from archive.outer_lambda_exp010_exp011_legacy import (
        reweight_exp011_umbrella_reports as _legacy_implementation,
    )

    return _legacy_implementation(reports, target_hamiltonian_id=target_hamiltonian_id, minimum_neighbor_overlap=minimum_neighbor_overlap)


def _run_cli_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.command in {"exp011-umbrella-sample", "exp011-reweight-umbrella"}:
        protocol = _cli_read_json_mapping(args.protocol, "EXP-011 protocol")
        stored_hash = protocol.get("protocol_sha256")
        protocol_core = dict(protocol)
        protocol_core.pop("protocol_sha256", None)
        if (
            protocol.get("protocol_type") != "outer_lambda_exp011_preregistration"
            or not isinstance(stored_hash, str)
            or stable_payload_sha256(protocol_core) != stored_hash
        ):
            raise NeuralPathIntegrityError("EXP-011 protocol 类型或内部 SHA-256 不匹配")
        target = protocol.get("target")
        if not isinstance(target, Mapping):
            raise NeuralPathConfigError("EXP-011 protocol 缺少 target")

        if args.command == "exp011-umbrella-sample":
            manifest = _cli_read_json_mapping(args.manifest, "EXP-011 manifest")
            if (
                manifest.get("status") != "frozen_for_exp011_complete_mm_pmf"
                or manifest.get("target_experiment") != "EXP-011"
                or manifest.get("production_approval") is not False
            ):
                raise NeuralPathConfigError("manifest 不是冻结的 EXP-011 非生产 CV")
            torsion = manifest.get("primary_slow_variable", {}).get("atom_indices")
            if torsion != target.get("atom_indices"):
                raise NeuralPathIntegrityError("manifest torsion 与 EXP-011 protocol 不一致")
            report = run_hard_window_scratch_trajectory(
                args.baseline_root,
                args.output_dir,
                window_index=0,
                initial_trajectory_path=args.initial_trajectory,
                burnin_steps=args.burnin_steps,
                sampling_steps=args.sampling_steps,
                report_interval_steps=args.report_interval_steps,
                platform_name=args.platform,
                random_seed=args.seed,
                umbrella_torsion_atom_indices=torsion,
                umbrella_center_degrees=args.center_degrees,
                umbrella_force_constant_kj_mol_radian2=(
                    args.force_constant_kj_mol_radian2
                ),
                umbrella_run_id=args.run_id,
                minimize_max_iterations=args.minimize_max_iterations,
            )
            report.update(
                {
                    "command": "exp011-umbrella-sample",
                    "protocol_sha256": stored_hash,
                    "protocol_file_sha256": sha256_file(args.protocol),
                    "manifest_file_sha256": sha256_file(args.manifest),
                }
            )
            return report

        reports = [
            _cli_read_json_mapping(path, f"umbrella input[{index}]")
            for index, path in enumerate(args.input)
        ]
        report = reweight_exp011_umbrella_reports(
            reports,
            target_hamiltonian_id=target.get("target_hamiltonian_id"),
            minimum_neighbor_overlap=args.minimum_neighbor_overlap,
        )
        dataset = report.pop("dataset")
        dataset.update(
            {
                "protocol_sha256": stored_hash,
                "source_report_file_sha256": [sha256_file(path) for path in args.input],
            }
        )
        _cli_write_json(dataset, args.output_dataset)
        report.update(
            {
                "ok": True,
                "command": "exp011-reweight-umbrella",
                "protocol_sha256": stored_hash,
                "output_dataset": str(Path(args.output_dataset).expanduser().resolve()),
                "output_dataset_sha256": sha256_file(args.output_dataset),
            }
        )
        return report

    if args.command in {"exp011-coverage", "exp011-fit-pmf"}:
        protocol = _cli_read_json_mapping(args.protocol, "EXP-011 protocol")
        stored_hash = protocol.get("protocol_sha256")
        protocol_core = dict(protocol)
        protocol_core.pop("protocol_sha256", None)
        if (
            protocol.get("protocol_type") != "outer_lambda_exp011_preregistration"
            or not isinstance(stored_hash, str)
            or stable_payload_sha256(protocol_core) != stored_hash
        ):
            raise NeuralPathIntegrityError(
                "EXP-011 protocol 类型或内部 SHA-256 不匹配"
            )
        if protocol.get("status") != "PREREGISTERED_NOT_STARTED":
            raise NeuralPathConfigError(
                "EXP-011 protocol status 必须为 PREREGISTERED_NOT_STARTED"
            )

        if args.command == "exp011-fit-pmf":
            dataset = _cli_read_json_mapping(args.dataset, "EXP-011 dataset")
            report = fit_exp011_reweighted_periodic_pmf(dataset, protocol)
            report.update(
                {
                    "ok": True,
                    "command": "exp011-fit-pmf",
                    "protocol_path": str(Path(args.protocol).expanduser().resolve()),
                    "protocol_file_sha256": sha256_file(args.protocol),
                    "protocol_sha256": stored_hash,
                    "dataset_path": str(Path(args.dataset).expanduser().resolve()),
                    "dataset_file_sha256": sha256_file(args.dataset),
                }
            )
            return report

        topology_path = Path(args.topology).expanduser()
        manifest_path = Path(args.manifest).expanduser()
        if not topology_path.is_file() or not manifest_path.is_file():
            raise NeuralPathConfigError("EXP-011 topology/manifest 文件不存在")
        manifest = _cli_read_json_mapping(manifest_path, "slow-variable manifest")
        primary = manifest.get("primary_slow_variable")
        atom_indices = primary.get("atom_indices") if isinstance(primary, Mapping) else None
        if not isinstance(atom_indices, list) or len(atom_indices) != 4:
            raise NeuralPathConfigError("slow-variable manifest 缺少 primary 四原子 torsion")
        run_angles = {}
        trajectory_records = []
        frozen_g = None
        if bool(args.trajectory) == bool(args.screen_report):
            raise NeuralPathConfigError(
                "exp011-coverage 必须且只能提供 trajectory 或 screen-report"
            )
        if args.screen_report:
            frozen_g = {}
            primary_id = primary.get("stable_id")
            for run_index, raw_report_path in enumerate(args.screen_report, start=1):
                report_path = Path(raw_report_path).expanduser()
                screen = _cli_read_json_mapping(report_path, "screen-report")
                candidates = screen.get("periodic_torsion_candidates")
                if not isinstance(candidates, list):
                    raise NeuralPathConfigError("screen-report 缺少 periodic candidates")
                candidate = next(
                    (item for item in candidates if item.get("stable_id") == primary_id),
                    None,
                )
                if candidate is None:
                    raise NeuralPathConfigError("screen-report 缺少冻结 primary torsion")
                histogram = candidate.get("torsion", {}).get("histogram", {})
                counts = histogram.get("counts")
                edges = histogram.get("bin_edges_degrees")
                if (
                    not isinstance(counts, list)
                    or not isinstance(edges, list)
                    or len(edges) != len(counts) + 1
                    or len(counts) != int(protocol["coverage"]["bins"])
                ):
                    raise NeuralPathConfigError("screen-report histogram 与冻结 bins 不一致")
                angles = []
                for bin_index, count in enumerate(counts):
                    center = 0.5 * (float(edges[bin_index]) + float(edges[bin_index + 1]))
                    angles.extend([center] * int(count))
                run_id = f"run{run_index}:{report_path.parent.name}"
                run_angles[run_id] = angles
                frozen_g[run_id] = _finite_float(
                    candidate.get("periodic_statistical_inefficiency"),
                    "screen-report periodic g",
                )
                trajectory_records.append(
                    {
                        "run_id": run_id,
                        "screen_report_path": str(report_path.resolve()),
                        "screen_report_sha256": sha256_file(report_path),
                        "selected_frame_count": len(angles),
                        "statistics_source": "precomputed_exact_histogram_and_periodic_g",
                    }
                )
        else:
            try:
                import mdtraj as md
            except ImportError as exc:
                raise NeuralPathConfigError("trajectory 模式需要安装 mdtraj") from exc
            trajectories = [Path(path).expanduser() for path in args.trajectory]
            if any(not path.is_file() for path in trajectories):
                raise NeuralPathConfigError("EXP-011 trajectory 文件不存在")
            for run_index, trajectory_path in enumerate(trajectories, start=1):
                with md.open(str(trajectory_path)) as handle:
                    frame_count = len(handle)
                frame_indices = (
                    tuple(range(frame_count))
                    if args.frames == "all"
                    else _resolve_trajectory_frame_spec(args.frames, frame_count)
                )
                selected = set(frame_indices)
                angles = []
                offset = 0
                for chunk in md.iterload(
                    str(trajectory_path), top=str(topology_path), chunk=250
                ):
                    values = md.compute_dihedrals(
                        chunk, [list(map(int, atom_indices))], periodic=True
                    )[:, 0]
                    for local_index, value in enumerate(values):
                        if offset + local_index in selected:
                            angles.append(math.degrees(float(value)))
                    offset += len(chunk)
                run_id = f"run{run_index}:{trajectory_path.parent.parent.name}"
                run_angles[run_id] = angles
                trajectory_records.append(
                    {
                        "run_id": run_id,
                        "path": str(trajectory_path.resolve()),
                        "sha256": sha256_file(trajectory_path),
                        "total_frame_count": frame_count,
                        "selected_frame_count": len(angles),
                    }
                )
        report = assess_exp011_periodic_coverage(
            run_angles,
            run_statistical_inefficiencies=frozen_g,
            **dict(protocol["coverage"]),
        )
        report.update(
            {
                "ok": True,
                "command": "exp011-coverage",
                "analysis_scope": "coverage_only_not_target_pmf_samples",
                "torsion_atom_indices": list(map(int, atom_indices)),
                "protocol_path": str(Path(args.protocol).expanduser().resolve()),
                "protocol_file_sha256": sha256_file(args.protocol),
                "protocol_sha256": stored_hash,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "topology_path": str(topology_path.resolve()),
                "topology_sha256": sha256_file(topology_path),
                "frame_spec": args.frames,
                "trajectories": trajectory_records,
                "pmf_prohibition": (
                    "这些 IBS mixture scratch 轨迹没有逐帧目标态权重；只能判定 CV 覆盖，"
                    "不得直接当作单一目标 Hamiltonian 的 PMF 样本"
                ),
            }
        )
        return report

    if args.command == "sample-hard-window-scratch":
        report = run_hard_window_scratch_trajectory(
            args.baseline_root,
            args.output_dir,
            window_index=args.window_index,
            initial_trajectory_path=args.initial_trajectory,
            burnin_steps=args.burnin_steps,
            sampling_steps=args.sampling_steps,
            report_interval_steps=args.report_interval_steps,
            platform_name=args.platform,
            random_seed=args.seed,
        )
        report["command"] = "sample-hard-window-scratch"
        return report



    if args.command == "screen-slow-variables":
        try:
            import mdtraj as md
        except ImportError as exc:
            raise NeuralPathConfigError(
                "screen-slow-variables 需要安装 mdtraj"
            ) from exc
        trajectory_path = Path(args.trajectory).expanduser()
        topology_path = Path(args.topology).expanduser()
        if not trajectory_path.is_file() or not topology_path.is_file():
            raise NeuralPathConfigError("trajectory/topology 文件不存在")
        with md.open(str(trajectory_path)) as handle:
            frame_count = len(handle)
        if args.frames == "all":
            frame_indices = tuple(range(frame_count))
        else:
            frame_indices = _resolve_trajectory_frame_spec(
                args.frames, frame_count
            )
        if len(frame_indices) == 1:
            loaded = md.load_frame(
                str(trajectory_path),
                frame_indices[0],
                top=str(topology_path),
            )
        else:
            step = frame_indices[1] - frame_indices[0]
            arithmetic = (
                step > 0
                and tuple(
                    range(
                        frame_indices[0],
                        frame_indices[-1] + step,
                        step,
                    )
                )
                == tuple(frame_indices)
                and frame_indices[0] % step == 0
            )
            if arithmetic:
                strided = md.load(
                    str(trajectory_path),
                    top=str(topology_path),
                    stride=step,
                )
                selected_positions = [
                    index // step for index in frame_indices
                ]
                loaded = strided[selected_positions]
            else:
                full_trajectory = md.load(
                    str(trajectory_path), top=str(topology_path)
                )
                loaded = full_trajectory[list(frame_indices)]
        ligand_payload = _cli_read_json_mapping(
            args.ligand_indices, "ligand-indices"
        )
        ligand_indices = ligand_payload.get("ligand_indices")
        if not isinstance(ligand_indices, list) or not ligand_indices:
            raise NeuralPathConfigError(
                "ligand-indices JSON 缺少非空 ligand_indices"
            )
        topology = loaded.topology
        bond_pairs = None
        system_sha = None
        if args.system_xml:
            system_path = Path(args.system_xml).expanduser()
            if not system_path.is_file():
                raise NeuralPathConfigError("system-xml 文件不存在")
            openmm, _ = _require_openmm()
            system = openmm.XmlSerializer.deserialize(
                system_path.read_text(encoding="utf-8")
            )
            bond_pairs = []
            for force_index in range(system.getNumForces()):
                force = system.getForce(force_index)
                if isinstance(force, openmm.HarmonicBondForce):
                    for bond_index in range(force.getNumBonds()):
                        left, right, _, _ = force.getBondParameters(bond_index)
                        bond_pairs.append([int(left), int(right)])
            system_sha = sha256_file(system_path)
        ligand_torsions = discover_ligand_rotatable_torsions(
            topology, ligand_indices, bond_pairs=bond_pairs
        )
        sidechain_torsions = discover_pocket_sidechain_chi1_torsions(
            topology,
            loaded.xyz[0].tolist(),
            ligand_indices,
            box_vectors_nm=(
                loaded.unitcell_vectors[0].tolist()
                if loaded.unitcell_vectors is not None
                else None
            ),
            pocket_cutoff_nm=args.pocket_cutoff_nm,
        )
        report = screen_periodic_torsion_candidates(
            loaded.xyz.tolist(),
            (
                loaded.unitcell_vectors.tolist()
                if loaded.unitcell_vectors is not None
                else None
            ),
            ligand_torsions + sidechain_torsions,
        )
        report["hydration_candidate"] = screen_ligand_hydration_coordination(
            loaded.xyz.tolist(),
            (
                loaded.unitcell_vectors.tolist()
                if loaded.unitcell_vectors is not None
                else None
            ),
            topology,
            ligand_indices,
            switching_distance_nm=args.hydration_switching_distance_nm,
            switching_power=args.hydration_switching_power,
        )
        report.update(
            {
                "ok": True,
                "command": "screen-slow-variables",
                "trajectory": str(trajectory_path.resolve()),
                "trajectory_sha256": sha256_file(trajectory_path),
                "topology": str(topology_path.resolve()),
                "topology_sha256": sha256_file(topology_path),
                "frame_indices": list(frame_indices),
                "ligand_rotatable_torsion_count": len(ligand_torsions),
                "pocket_sidechain_chi1_count": len(sidechain_torsions),
                "pocket_cutoff_nm": float(args.pocket_cutoff_nm),
                "system_xml": (
                    str(Path(args.system_xml).expanduser().resolve())
                    if args.system_xml
                    else None
                ),
                "system_xml_sha256": system_sha,
                "ligand_bond_source": (
                    "system_harmonic_bond_force"
                    if bond_pairs is not None
                    else "topology"
                ),
            }
        )
        return report

    if args.command == "compare-slow-variable-screens":
        reports = [
            _cli_read_json_mapping(path, f"input[{index}]")
            for index, path in enumerate(args.input)
        ]
        report = compare_slow_variable_screens(reports)
        report.update(
            {
                "ok": True,
                "command": "compare-slow-variable-screens",
                "inputs": [
                    {
                        "path": str(Path(path).expanduser().resolve()),
                        "sha256": sha256_file(path),
                    }
                    for path in args.input
                ],
            }
        )
        return report

    if args.command == "freeze-slow-variable":
        comparison = _cli_read_json_mapping(
            args.comparison, "comparison"
        )
        final_results = _cli_read_json_mapping(
            args.final_results, "final-results"
        )
        selected_window = select_wp0_difficult_window(final_results)[
            "selected_window"
        ]
        report = freeze_slow_variable_manifest(
            comparison,
            selected_window,
            replicate_rank=args.replicate_rank,
            experiment_id=args.experiment,
        )
        report.update(
            {
                "ok": True,
                "command": "freeze-slow-variable",
                "comparison_path": str(
                    Path(args.comparison).expanduser().resolve()
                ),
                "comparison_sha256": sha256_file(args.comparison),
                "final_results_path": str(
                    Path(args.final_results).expanduser().resolve()
                ),
                "final_results_sha256": sha256_file(args.final_results),
            }
        )
        return report

    if args.command == "exp010-prepare-selection":
        try:
            import mdtraj as md
        except ImportError as exc:
            raise NeuralPathConfigError(
                "exp010-prepare-selection 需要安装 mdtraj"
            ) from exc
        topology_path = Path(args.topology).expanduser()
        if not topology_path.is_file():
            raise NeuralPathConfigError("exp010 topology 文件不存在")
        topology = md.load(str(topology_path)).topology
        source = _cli_read_json_mapping(
            args.selection_meta, "selection-meta"
        )
        selection = build_exp010_protein_only_selection(source, topology)
        _cli_write_json(selection, args.output_selection_meta)
        return {
            "ok": True,
            "command": "exp010-prepare-selection",
            "report_type": "outer_lambda_exp010_selection_preparation",
            "report_version": 1,
            "source_selection_meta": str(
                Path(args.selection_meta).expanduser().resolve()
            ),
            "source_selection_meta_sha256": sha256_file(args.selection_meta),
            "output_selection_meta": str(
                Path(args.output_selection_meta).expanduser().resolve()
            ),
            "output_selection_meta_sha256": sha256_file(
                args.output_selection_meta
            ),
            "selection_policy": selection[
                "outer_lambda_exp010_selection_policy"
            ],
            "selection_protocol_sha256": selection["selection_sha256"],
        }

    if args.command == "exp010-label":
        try:
            import mdtraj as md
        except ImportError as exc:
            raise NeuralPathConfigError("exp010-label 需要安装 mdtraj") from exc
        controller = load_neural_path_config(
            args.config, verify_basis_files=True
        )
        if not controller.enabled or controller.basis_count != 1:
            raise NeuralPathConfigError("exp010-label 要求启用且严格 M=1")
        basis = controller.bases[0]
        if basis.backend != "existing_openmmml" or not basis.model_name:
            raise NeuralPathConfigError(
                "exp010-label 要求 existing_openmmml MACE/ORB basis"
            )
        manifest = _cli_read_json_mapping(args.manifest, "manifest")
        stored_manifest_sha = manifest.get("manifest_sha256")
        manifest_core = dict(manifest)
        for key in (
            "manifest_sha256",
            "ok",
            "command",
            "comparison_path",
            "comparison_sha256",
            "final_results_path",
            "final_results_sha256",
        ):
            manifest_core.pop(key, None)
        if (
            not isinstance(stored_manifest_sha, str)
            or stable_payload_sha256(manifest_core) != stored_manifest_sha
        ):
            raise NeuralPathIntegrityError(
                "slow-variable manifest 内部 SHA-256 不匹配"
            )
        selection = _cli_read_json_mapping(
            args.selection_meta, "selection-meta"
        )
        ligand_indices = selection.get("ligand_indices")
        environment_indices = selection.get("env_indices")
        if not isinstance(ligand_indices, list) or not isinstance(
            environment_indices, list
        ):
            raise NeuralPathConfigError(
                "selection-meta 缺少 ligand_indices/env_indices"
            )
        if set(basis.atom_indices()) != set(ligand_indices).union(
            environment_indices
        ):
            raise NeuralPathConfigError(
                "配置 atom selection 与 selection-meta 不一致"
            )
        topology_path = Path(args.topology).expanduser()
        if not topology_path.is_file():
            raise NeuralPathConfigError("exp010 topology 文件不存在")
        trajectory_paths = [
            Path(path).expanduser() for path in args.trajectory
        ]
        if any(not path.is_file() for path in trajectory_paths):
            raise NeuralPathConfigError("exp010 trajectory 文件不存在")
        reference = md.load_frame(
            str(trajectory_paths[0]), 0, top=str(topology_path)
        )
        atomic_numbers = []
        for atom in reference.topology.atoms:
            if atom.element is None:
                raise NeuralPathConfigError(
                    f"topology atom {atom.index} 缺少元素"
                )
            atomic_numbers.append(int(atom.element.atomic_number))
        run_specs = []
        for run_index, path in enumerate(trajectory_paths):
            with md.open(str(path)) as handle:
                frame_count = len(handle)
            indices = _resolve_trajectory_frame_spec(args.frames, frame_count)
            run_specs.append(
                {
                    "run_id": f"run{run_index + 1}:{path.parent.name}",
                    "path": path,
                    "frame_count": frame_count,
                    "frame_indices": indices,
                }
            )

        support_evaluations = []
        exclusion_limit = _finite_float(
            args.max_support_exclusion_fraction,
            "max_support_exclusion_fraction",
        )
        if not 0.0 <= exclusion_limit < 1.0:
            raise NeuralPathConfigError(
                "max_support_exclusion_fraction 必须位于 [0,1)"
            )

        def frame_records():
            for run_spec in run_specs:
                indices = run_spec["frame_indices"]
                step = indices[1] - indices[0] if len(indices) > 1 else 0
                arithmetic = (
                    step > 0
                    and tuple(range(indices[0], indices[-1] + step, step))
                    == tuple(indices)
                    and indices[0] % step == 0
                )
                strided = (
                    md.load(
                        str(run_spec["path"]),
                        top=str(topology_path),
                        stride=step,
                    )
                    if arithmetic
                    else None
                )
                for frame_index in indices:
                    frame = (
                        strided[frame_index // step]
                        if strided is not None
                        else md.load_frame(
                            str(run_spec["path"]),
                            frame_index,
                            top=str(topology_path),
                        )
                    )
                    box_vectors = (
                        frame.unitcell_vectors[0].tolist()
                        if frame.unitcell_vectors is not None
                        else None
                    )
                    support = controller.evaluate_support_domains(
                        frame.xyz[0].tolist(),
                        box_vectors_nm=box_vectors,
                    )
                    supported = all(item.supported for item in support)
                    support_record = {
                        "run_id": run_spec["run_id"],
                        "frame_index": frame_index,
                        "supported": supported,
                        "included_in_teacher_dataset": supported,
                        "details": [
                            item.payload() for item in support
                        ],
                    }
                    support_evaluations.append(support_record)
                    if not supported:
                        if args.support_violation_policy == "reject":
                            raise NeuralPathConfigError(
                                "EXP-010 source frame 超出冻结 MACE 支持域: "
                                f"{run_spec['run_id']} frame={frame_index}"
                            )
                        continue
                    yield {
                        "run_id": run_spec["run_id"],
                        "frame_index": frame_index,
                        "positions_nm": frame.xyz[0].tolist(),
                        "box_vectors_nm": box_vectors,
                    }

        with ExistingOrbMaceBasisAdapter(
            model_name=basis.model_name, device=args.device
        ) as adapter:
            report = build_exp010_teacher_dataset(
                adapter,
                frame_records(),
                manifest,
                ligand_indices=ligand_indices,
                environment_indices=environment_indices,
                atomic_numbers=atomic_numbers,
                energy_offset_kj_mol=(
                    basis.energy_offset_kj_mol
                    if args.energy_offset_mode == "config"
                    else None
                ),
                include_secondary=not args.primary_only,
            )
        source_support_violation_count = sum(
            not item["supported"] for item in support_evaluations
        )
        source_frame_count = len(support_evaluations)
        exclusion_fraction = (
            source_support_violation_count / source_frame_count
            if source_frame_count
            else 1.0
        )
        safety_violation_count = 0
        if controller.safety is not None:
            for sample in report["samples"]:
                if (
                    abs(sample["teacher_centered_energy_kj_mol"])
                    > controller.safety.max_abs_basis_energy_kj_mol
                    or sample["teacher_max_force_kj_mol_nm"]
                    > controller.safety.max_force_norm_kj_mol_nm
                ):
                    safety_violation_count += 1
        report["support_domain_violation_count"] = 0
        report["source_support_domain_violation_count"] = (
            source_support_violation_count
        )
        report["source_frame_count"] = source_frame_count
        report["support_exclusion_fraction"] = exclusion_fraction
        report["max_support_exclusion_fraction"] = exclusion_limit
        report["support_violation_policy"] = args.support_violation_policy
        report["safety_violation_count"] = safety_violation_count
        report["qualified_for_fit"] = (
            safety_violation_count == 0
            and exclusion_fraction <= exclusion_limit
        )
        report["support_domain"] = support_evaluations
        report.update(
            {
                "ok": True,
                "command": "exp010-label",
                "config_path": str(Path(args.config).expanduser().resolve()),
                "config_sha256": sha256_file(args.config),
                "controller_protocol_sha256": controller.protocol_sha256(),
                "teacher_model_sha256": basis.sha256,
                "atom_selection_sha256": basis.atom_indices_sha256,
                "slow_variable_manifest_path": str(
                    Path(args.manifest).expanduser().resolve()
                ),
                "slow_variable_manifest_file_sha256": sha256_file(
                    args.manifest
                ),
                "slow_variable_manifest_protocol_sha256": stored_manifest_sha,
                "topology_path": str(topology_path.resolve()),
                "topology_sha256": sha256_file(topology_path),
                "selection_meta_path": str(
                    Path(args.selection_meta).expanduser().resolve()
                ),
                "selection_meta_sha256": sha256_file(args.selection_meta),
                "frame_spec": args.frames,
                "trajectories": [
                    {
                        "run_id": spec["run_id"],
                        "path": str(spec["path"].resolve()),
                        "sha256": sha256_file(spec["path"]),
                        "total_frame_count": spec["frame_count"],
                        "selected_frame_indices": list(spec["frame_indices"]),
                    }
                    for spec in run_specs
                ],
            }
        )
        return report

    if args.command == "exp010-fit":
        dataset = _cli_read_json_mapping(args.dataset, "dataset")
        if dataset.get("qualified_for_fit") is not True:
            raise NeuralPathConfigError(
                "teacher dataset 未通过 support/safety 门，拒绝拟合"
            )
        report = fit_periodic_fourier_distillation(
            dataset,
            dimensions=args.dimensions,
            order=args.order,
            ridge=args.ridge,
            conditional_bins=args.conditional_bins,
        )
        dataset_sha = sha256_file(args.dataset)
        model = report["model"]
        model.pop("model_sha256", None)
        model["training_dataset_sha256"] = dataset_sha
        model["teacher_model_sha256"] = dataset.get(
            "teacher_model_sha256"
        )
        model["slow_variable_manifest_protocol_sha256"] = dataset.get(
            "slow_variable_manifest_protocol_sha256"
        )
        model["model_sha256"] = stable_payload_sha256(model)
        report.update(
            {
                "ok": True,
                "command": "exp010-fit",
                "dataset_path": str(
                    Path(args.dataset).expanduser().resolve()
                ),
                "dataset_sha256": dataset_sha,
            }
        )
        return report

    if args.command == "wp0-select":
        report = build_wp0_selection_report(
            final_results_path=args.final_results,
            topology_path=args.topology,
            trajectory_paths=args.trajectory,
            torsion_atom_indices=args.torsion_indices,
            slow_variable_name=args.slow_variable_name,
        )
        report["ok"] = True
        report["command"] = "wp0-select"
        return report

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
            min_pair_distance_nm=args.min_pair_distance_nm,
            max_pair_distance_nm=args.max_pair_distance_nm,
            max_radius_of_gyration_nm=args.max_radius_of_gyration_nm,
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
            box_vectors_by_frame_nm=(
                existing_input.get("box_vectors_by_frame_nm")
                or (
                    [existing_input["box_vectors_nm"]] * len(frames)
                    if "box_vectors_nm" in existing_input
                    else None
                )
            ),
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
            box_vectors_by_frame_nm=[
                frame.unitcell_vectors[0].tolist()
                for frame in loaded_frames
            ],
        )
        report["ok"] = True
        report["command"] = "label-trajectory"
        report["trajectory"] = str(trajectory_path.resolve())
        report["topology"] = str(topology_path.resolve())
        report["trajectory_frame_count"] = n_trajectory_frames
        report["frame_indices"] = list(frame_indices)
        return report

    if args.command == "mace-mts-qualification":
        try:
            import mdtraj as md
        except ImportError as exc:
            raise NeuralPathConfigError(
                "mace-mts-qualification 需要安装 mdtraj"
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
                    "mace-mts-qualification --frame 必须只选择一个 frame"
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
                f"mace-mts-qualification 无法读取轨迹: {exc}"
            ) from exc
        if frame.unitcell_vectors is None:
            raise NeuralPathConfigError("轨迹 frame 缺少周期盒向量")
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
        arm_reports = []
        for ratio in (1, 2, 4):
            arm_reports.append(
                run_mace_decomposition_mts_arm(
                    controller,
                    base_system,
                    atomic_numbers=atomic_numbers,
                    ligand_indices=selection_meta.get("ligand_indices"),
                    environment_indices=selection_meta.get("env_indices"),
                    positions_nm=frame.xyz[0].tolist(),
                    box_vectors_nm=frame.unitcell_vectors[0].tolist(),
                    torsion_atom_indices=args.torsion_indices,
                    mts_ratio=ratio,
                    lambda_value=args.lambda_value,
                    n_inner_steps=args.inner_steps,
                    report_interval_inner_steps=(
                        args.report_interval_inner_steps
                    ),
                    inner_timestep_fs=args.inner_timestep_fs,
                    temperature_kelvin=args.temperature_k,
                    friction_per_ps=args.friction_per_ps,
                    device=args.device,
                    platform_name=args.platform,
                    random_seed=args.seed,
                    required_coefficient=0.09,
                )
            )
        report = assess_mace_mts_matrix(
            arm_reports,
            minimum_n4_ns_per_day=args.minimum_n4_ns_per_day,
        )
        report.update(
            {
                "ok": True,
                "command": "mace-mts-qualification",
                "system_xml": str(system_xml_path.resolve()),
                "trajectory": str(trajectory_path.resolve()),
                "topology": str(topology_path.resolve()),
                "frame_index": frame_indices[0],
                "torsion_atom_indices": list(args.torsion_indices),
            }
        )
        return report

    if args.command in {"mace-nvt-smoke", "mace-nvt-qualification"}:
        command_name = args.command
        try:
            import mdtraj as md
        except ImportError as exc:
            raise NeuralPathConfigError(
                f"{command_name} 需要安装 mdtraj"
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
                    f"{command_name} --frame 必须只选择一个 frame"
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
                f"{command_name} 无法读取轨迹: {exc}"
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
        nvt_runner = (
            run_mace_decomposition_nvt_smoke
            if command_name == "mace-nvt-smoke"
            else run_mace_decomposition_nvt
        )
        report = nvt_runner(
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
        run_metadata = {
            "system_xml": str(system_xml_path.resolve()),
            "trajectory": str(trajectory_path.resolve()),
            "topology": str(topology_path.resolve()),
            "frame_index": frame_indices[0],
        }
        report.update(run_metadata)
        if command_name == "mace-nvt-smoke":
            report["ok"] = True
            report["command"] = command_name
            return report
        qualification = assess_mace_nvt_qualification(
            report,
            minimum_steps=args.minimum_steps,
            max_path_force_kj_mol_nm=args.max_path_force_kj_mol_nm,
            max_energy_closure_error_kj_mol=(
                args.max_energy_closure_error_kj_mol
            ),
            max_integration_seconds_per_step=(
                args.max_integration_seconds_per_step
            ),
        )
        qualification["ok"] = True
        qualification["command"] = command_name
        return qualification

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
        if (
            args.command
            in {"mace-nvt-qualification", "mace-mts-qualification"}
            and result.get("qualified") is not True
        ):
            return 1
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
    "MaceDecompositionBasisPythonComputation",
    "OuterLambdaIBSBiasForce",
    "IBSEnergyFrame",
    "IBSEnergyLedger",
    "IBSSamplerNeuralPathAdapter",
    "compose_ibs_energy_frame",
    "OpenMMPathEvaluation",
    "build_torchforce_from_spec",
    "build_openmm_outer_lambda_force",
    "build_torchforce_outer_lambda_force",
    "build_mace_decomposition_python_force",
    "build_mace_decomposition_basis_python_force",
    "evaluate_outer_lambda_force_group_states",
    "run_mace_decomposition_nvt",
    "run_mace_decomposition_nvt_smoke",
    "assess_mace_nvt_qualification",
    "run_mace_decomposition_mts_arm",
    "assess_mace_mts_matrix",
    "periodic_dihedral_degrees",
    "classify_torsion_basin",
    "analyze_periodic_torsion_series",
    "discover_ligand_rotatable_torsions",
    "discover_pocket_sidechain_chi1_torsions",
    "screen_periodic_torsion_candidates",
    "screen_ligand_hydration_coordination",
    "compare_slow_variable_screens",
    "freeze_slow_variable_manifest",
    "torsion_coordinate_gradient_radians",
    "build_exp010_protein_only_selection",
    "project_force_onto_torsion",
    "build_exp010_teacher_dataset",
    "fit_periodic_fourier_distillation",
    "assess_exp011_periodic_coverage",
    "fit_exp011_reweighted_periodic_pmf",
    "build_exp011_periodic_umbrella_force",
    "reweight_exp011_umbrella_reports",
    "build_periodic_fourier_openmm_force",
    "run_hard_window_scratch_trajectory",
    "select_wp0_difficult_window",
    "build_wp0_selection_report",
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

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


# ORB/MACE 分解 Context 缓存的容量上限。每条 entry 持有 3 个带模型的
# OpenMM Context，所以这个数直接决定显存占用上限；调大之前先量一个 entry 多大。
_ORB_DECOMPOSITION_CACHE_MAX_ENTRIES = 4


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



#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RBFE（相对结合自由能）核心：数据契约、输入验证、ΔΔG 汇总。

设计依据：`docs/design/PLAN_rbfe_interface_and_implementation.md`。

## 当前实现进度：R0

计划 §8 的阶段表里，本文件目前只实现 **R0**：

    schema、验证接口、方向明确的结果汇总、（供 runrbfe.py validate 调用）

R0 的验收标准是「错误输入被拒绝；合成两腿数据的符号／单位／误差传播正确；
**不启动 GPU**」——本文件因此不 import openmm、不构建任何 System、不做任何采样。

**R1 起（原子映射、hybrid builder、采样、分析）一行都还没写。** 对应的函数在文件
末尾以 `NotImplementedError` 显式占位。计划 §4.1 明令：「不提供『尚未实现但返回
成功』的占位 sampler；接口准备完成与可运行科学计算是两个验收阶段。」——所以那些
函数**抛错**，不返回假数据。

## 依赖方向（不得违反）

    runrbfe.py -> rbfe_pipeline.py -> rbfe_core.py
                                   -> free_energy_engine.py

本模块**任何位置**都不 import `rbfe_pipeline`——那是反向依赖，会成环。

`abfe_core` / `ibs_engine` 则是**允许的、且是刻意的**：2026-09-03 用户明确
「直接 import `abfe_core` 复用，但不改 ABFE 一行代码」。R1b 的 softcore/排除表
与 R2 的 MBAR 数值求解都直接用 ABFE 那边久经生产验证的实现（计划 §6 也把
「独立的 MBAR 数值求解部分」列为可复用）。

约束只剩**位置**：这些 import（以及 openmm）一律**放在函数体内**，不上模块顶层。
理由是 R0/R1a 那两层的性质——「不 import openmm、不启动 GPU」——是它们上百条
测试的前提；顶层 import 会让 `import rbfe_core` 无条件拉进 openmm/pymbar，
把那个性质连同测试一起废掉。`test_core_has_no_module_level_openmm_or_abfe_import`
与 `test_lazy_imports_are_actually_lazy` 两条测试守着这件事。

## 为什么不能复用 ABFE 的汇总函数

计划 §3 用加粗写死了这条：ABFE 走的是 coupled→decoupled 去耦约定、最终采用
**solvent − complex**；RBFE 走 A→B 变换、采用 **complex − solvent**。

    ABFE:  ΔG_bind = ΔG_solvent - ΔG_complex + ΔG_APBS
    RBFE:  ΔΔG_bind(B-A) = ΔG_complex(A→B) - ΔG_solvent(A→B)

**符号相反。** 直接调用 ABFE 的汇总函数（或者只把名字改成 relative）会得到一个
符号翻转的结果，而且因为量级看着合理，很难在事后被发现。本模块因此自带一套完整
的汇总实现，一行都不从 ABFE 那边借。
"""

from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# 协议身份与常量
# ---------------------------------------------------------------------------

#: RBFE 数据契约的协议版本。进 edge_manifest / 结果指纹（R2 起）。
#: 任何改变 EdgeSpec 必填字段、验证判据或 ΔΔG 定义的改动都必须 +1。
RBFE_CORE_PROTOCOL_VERSION = 1

#: 变换方向的唯一约定：lambda=0 对应 A，lambda=1 对应 B（计划 §3）。
RBFE_DIRECTION = "A_to_B"

#: 支持的能量单位。内部统一 kJ/mol（与 ABFE 一致），kcal/mol 仅在输出层换算。
KJ_PER_MOL = "kJ/mol"
KCAL_PER_MOL = "kcal/mol"
SUPPORTED_ENERGY_UNITS = (KJ_PER_MOL, KCAL_PER_MOL)
_KJ_PER_KCAL = 4.184

#: 两条腿的相名。RBFE 是「同一个变换在两个环境里各做一次」。
PHASE_COMPLEX = "complex"
PHASE_SOLVENT = "solvent"
RBFE_PHASES = (PHASE_COMPLEX, PHASE_SOLVENT)


class RBFEValidationError(ValueError):
    """输入未通过 R0 验证。在创建任何生产 Context 之前抛出。"""


class RBFEUnsupportedTransformationError(RBFEValidationError):
    """变换类型落在首版范围之外（计划 §2）。

    🔑 这**不是**在说 RBFE 方法或别的软件不支持这类变换，只是本项目首版的范围限制
    （计划 §2 原话）。错误信息里必须保留这个区分，否则以后会有人拿它当"做不到"的
    结论引用。
    """


# ---------------------------------------------------------------------------
# 端点身份
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LigandEndpoint:
    """一个配体端点（A 或 B）的完整化学身份（计划 §5.1）。

    这些字段不是"最好有"，是**身份**：两次运行只要其中任何一项不同，产物就不可
    互换、缓存不可复用。`input_sha256` 让"同一个文件被换过内容"也能被发现——
    只比路径是不够的。
    """

    name: str
    #: 带显式氢的化学结构（SMILES / InChI）。计划 §5.1 要求显式氢——隐式氢的
    #: SMILES 无法区分不同质子化态，而质子化态改变是首版明令拒绝的变换之一。
    structure: str
    #: 形式净电荷（整数）。首版要求 A、B 都是中性且相等，见 validate_edge。
    formal_charge: int
    #: 参数/结构输入文件路径及其内容哈希。
    input_path: str
    input_sha256: str
    #: 质子化态与立体化学身份。留空表示"未声明"，验证会拒绝——沉默的默认值正是
    #: 这类项目最容易出错的地方。
    protonation_state: str
    stereochemistry: str
    #: 部分电荷来源（如 "am1bcc" / "resp" / "from_input_topology"）。
    partial_charge_source: str

    def identity(self) -> dict:
        return {
            "name": self.name,
            "structure": self.structure,
            "formal_charge": int(self.formal_charge),
            "input_path": self.input_path,
            "input_sha256": self.input_sha256,
            "protonation_state": self.protonation_state,
            "stereochemistry": self.stereochemistry,
            "partial_charge_source": self.partial_charge_source,
        }


@dataclass(frozen=True)
class EnvironmentSpec:
    """受体／环境身份（计划 §5.1）。"""

    receptor_name: str
    receptor_path: str
    receptor_sha256: str
    force_field: str
    water_model: str
    ion_model: str
    #: 首版拒绝膜体系（计划 §2）。显式声明而不是靠猜文件名。
    is_membrane: bool = False


@dataclass(frozen=True)
class ProtocolSpec:
    """采样与系综协议（计划 §2：显式配置，不沿用 ABFE 的隐藏默认值）。"""

    temperature_kelvin: float
    pressure_bar: Optional[float]
    n_lambda_states: int
    n_steps_per_state: int
    seed: int
    #: 显式记录，而不是从 ABFE 的 λ 调度协议继承——计划 §6 明确 ABFE 的 λ 调度
    #: 协议版本不可直接复用。
    lambda_schedule_name: str


@dataclass(frozen=True)
class EdgeSpec:
    """一条 A→B 边的完整输入（计划 §5.1）。"""

    edge_id: str
    ligand_a: LigandEndpoint
    ligand_b: LigandEndpoint
    environment: EnvironmentSpec
    protocol: ProtocolSpec
    output_dir: str
    energy_unit: str = KJ_PER_MOL
    protocol_version: int = RBFE_CORE_PROTOCOL_VERSION

    def manifest(self) -> dict:
        """写进 `edge_manifest.json` 的身份快照（计划 §7）。"""
        return {
            "edge_id": self.edge_id,
            "direction": RBFE_DIRECTION,
            "rbfe_core_protocol_version": int(self.protocol_version),
            "energy_unit": self.energy_unit,
            "ligand_A": self.ligand_a.identity(),
            "ligand_B": self.ligand_b.identity(),
            "environment": {
                "receptor_name": self.environment.receptor_name,
                "receptor_path": self.environment.receptor_path,
                "receptor_sha256": self.environment.receptor_sha256,
                "force_field": self.environment.force_field,
                "water_model": self.environment.water_model,
                "ion_model": self.environment.ion_model,
                "is_membrane": bool(self.environment.is_membrane),
            },
            "protocol": {
                "temperature_kelvin": float(self.protocol.temperature_kelvin),
                "pressure_bar": self.protocol.pressure_bar,
                "n_lambda_states": int(self.protocol.n_lambda_states),
                "n_steps_per_state": int(self.protocol.n_steps_per_state),
                "seed": int(self.protocol.seed),
                "lambda_schedule_name": self.protocol.lambda_schedule_name,
            },
            "output_dir": self.output_dir,
        }


# ---------------------------------------------------------------------------
# R0 验证
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """`validate_edge` 的结果。`errors` 非空即拒绝。"""

    edge_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: 本次**没有**检查的东西。诚实记录，免得"验证通过"被误读成"全都查过了"。
    unchecked: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise RBFEUnsupportedTransformationError(
                f"边 {self.edge_id} 未通过 R0 验证（{len(self.errors)} 项）：\n  - "
                + "\n  - ".join(self.errors)
            )

    def render(self) -> str:
        lines = [f"edge={self.edge_id}  结果={'PASS' if self.ok else 'REJECT'}"]
        for tag, items in (("错误", self.errors), ("警告", self.warnings), ("未检查", self.unchecked)):
            for item in items:
                lines.append(f"  [{tag}] {item}")
        return "\n".join(lines)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _check_endpoint(ep: LigandEndpoint, label: str, report: ValidationReport) -> None:
    if not ep.name.strip():
        report.errors.append(f"配体 {label}：name 为空")
    if not ep.structure.strip():
        report.errors.append(f"配体 {label}：structure（带显式氢的 SMILES/InChI）为空")
    elif "[H]" not in ep.structure and "H" not in ep.structure:
        # 弱判据，只给警告——InChI 的氢层写法不同，硬拒会误伤。
        report.warnings.append(
            f"配体 {label}：structure 里看不到显式氢，确认它不是隐式氢 SMILES"
            "（隐式氢无法区分质子化态）"
        )
    if not _SHA256_RE.match(ep.input_sha256 or ""):
        report.errors.append(
            f"配体 {label}：input_sha256 不是 64 位小写十六进制"
            "（只比路径不够——同一路径的文件内容可能已被换掉）"
        )
    if not ep.protonation_state.strip():
        report.errors.append(f"配体 {label}：protonation_state 未声明")
    if not ep.stereochemistry.strip():
        report.errors.append(f"配体 {label}：stereochemistry 未声明")
    if not ep.partial_charge_source.strip():
        report.errors.append(f"配体 {label}：partial_charge_source 未声明")


def validate_edge(
    spec: EdgeSpec,
    *,
    mapping: Optional["AtomMapping"] = None,
    graph_a: Optional["MolecularGraph"] = None,
    graph_b: Optional["MolecularGraph"] = None,
) -> ValidationReport:
    """R0 验证：把首版范围外的输入在**创建任何 Context 之前**拒掉（计划 §2）。

    首版明令拒绝的变换（计划 §2 逐条）：净电荷变化、带电配体、膜体系、质子化／
    互变异构改变。环断裂／闭合、环尺寸变化、手性反转、映射元素改变、共价配体
    这几类需要原子映射才能判定——**不给映射就查不了**，本函数把它们列进
    `unchecked`，不假装已经查过。

    给了 `mapping` + `graph_a` + `graph_b`（R1a 的产物）时，这几条里能查的会被
    真正查掉：`validate_mapping` 的错误与警告并进本报告，对应的 `unchecked`
    条目随之消失。三个参数必须**一起**给——只给一部分是配置错误，直接报错，
    不悄悄退回"没查"。
    """
    evidence = [mapping is not None, graph_a is not None, graph_b is not None]
    if any(evidence) and not all(evidence):
        raise RBFEValidationError(
            "validate_edge 的 mapping / graph_a / graph_b 必须一起给或一起不给："
            f"收到 mapping={mapping is not None}、graph_a={graph_a is not None}、"
            f"graph_b={graph_b is not None}"
        )
    has_mapping = all(evidence)
    report = ValidationReport(edge_id=spec.edge_id)

    if not spec.edge_id.strip():
        report.errors.append("edge_id 为空")

    if spec.energy_unit not in SUPPORTED_ENERGY_UNITS:
        report.errors.append(
            f"energy_unit={spec.energy_unit!r} 不支持；只接受 {SUPPORTED_ENERGY_UNITS}"
        )

    _check_endpoint(spec.ligand_a, "A", report)
    _check_endpoint(spec.ligand_b, "B", report)

    if spec.ligand_a.name == spec.ligand_b.name:
        report.errors.append(
            f"ligand_A 与 ligand_B 同名（{spec.ligand_a.name!r}）——"
            "A→A 自变换是 R3 的验收用例，不是一条可以直接跑的生产边"
        )

    # 🔑 计划 §2：首版拒绝净电荷变化，以及尚未验证的同电荷带电配体路线。
    # 两条是分开的判据，不能合并成一条"电荷不等就拒绝"。
    qa, qb = int(spec.ligand_a.formal_charge), int(spec.ligand_b.formal_charge)
    if qa != qb:
        report.errors.append(
            f"净电荷变化 {qa} -> {qb}：首版拒绝。"
            "（这是本项目首版的范围限制，不是 RBFE 方法本身不支持。）"
        )
    elif qa != 0:
        report.errors.append(
            f"A、B 净电荷相等但非中性（{qa}）：同电荷带电配体路线首版尚未验证，拒绝。"
            "（同上，是范围限制不是方法限制。）"
        )

    if spec.environment.is_membrane:
        report.errors.append("膜体系：首版拒绝（计划 §2）")

    if spec.ligand_a.protonation_state != spec.ligand_b.protonation_state:
        report.errors.append(
            f"质子化态改变（{spec.ligand_a.protonation_state!r} -> "
            f"{spec.ligand_b.protonation_state!r}）：首版拒绝"
        )

    # 协议合法性
    p = spec.protocol
    if not (p.temperature_kelvin > 0):
        report.errors.append(f"temperature_kelvin 必须为正：收到 {p.temperature_kelvin}")
    if p.pressure_bar is not None and not (p.pressure_bar > 0):
        report.errors.append(f"pressure_bar 若给出必须为正：收到 {p.pressure_bar}")
    if p.n_lambda_states < 2:
        report.errors.append(
            f"n_lambda_states 至少为 2（端点 A 与 B）：收到 {p.n_lambda_states}"
        )
    if p.n_steps_per_state < 1:
        report.errors.append(f"n_steps_per_state 必须为正：收到 {p.n_steps_per_state}")
    if not p.lambda_schedule_name.strip():
        report.errors.append(
            "lambda_schedule_name 未声明——计划 §2 要求 λ 协议显式配置，"
            "不沿用 ABFE 的隐藏默认值"
        )

    # 🔑 诚实记录本阶段查不了的东西。
    if has_mapping:
        # R1a 的映射在手：环、元素、核心连通性这几条能真查了，剩下的仍然不能。
        mapping_report = validate_mapping(mapping, graph_a, graph_b, edge_id=spec.edge_id)
        report.errors.extend(mapping_report.errors)
        report.warnings.extend(mapping_report.warnings)
        report.unchecked.extend(mapping_report.unchecked)
        if mapping.n_atoms_a != graph_a.n_atoms or mapping.n_atoms_b != graph_b.n_atoms:
            report.errors.append(
                "mapping 与所给的图对不上（原子数不同）——这份映射不是从这两个图算出来的"
            )
    else:
        # 这几项全都需要原子映射，没有映射就既不能通过也不能拒绝，只能声明"没查"。
        report.unchecked.extend(
            [
                "环断裂/闭合、环尺寸变化（需要原子映射，R1）",
                "手性反转、映射元素改变（需要原子映射，R1）",
                "共价配体（需要拓扑连接性分析，R1）",
                "骨架或结合模式大幅改变（需要几何比较，R1）",
                "虚拟位点/约束/自定义 Force 是否在已验证 builder 范围内（需要建系，R1）",
                "互变异构状态改变（structure 字段无法单独判定，R1）",
            ]
        )

    return report


# ---------------------------------------------------------------------------
# 结果与 ΔΔG 汇总
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegResult:
    """一条腿的结果（计划 §4.1 的 `LegResult` 契约）。

    `delta_g` 的含义被方向锁死：**G(B) − G(A) 在该相中的值**。
    """

    phase: str
    edge_id: str
    ligand_a_name: str
    ligand_b_name: str
    delta_g: float
    stderr: float
    energy_unit: str
    uncertainty_method: str
    n_effective_samples: int
    quality_gate_passed: bool
    artifacts_fingerprint: str
    direction: str = RBFE_DIRECTION

    def __post_init__(self) -> None:
        if self.phase not in RBFE_PHASES:
            raise ValueError(f"phase 必须是 {RBFE_PHASES} 之一：收到 {self.phase!r}")
        if self.direction != RBFE_DIRECTION:
            raise ValueError(
                f"direction 必须是 {RBFE_DIRECTION!r}：收到 {self.direction!r}"
            )
        if self.energy_unit not in SUPPORTED_ENERGY_UNITS:
            raise ValueError(f"energy_unit 不支持：{self.energy_unit!r}")
        if not math.isfinite(self.delta_g):
            raise ValueError(f"delta_g 非有限值：{self.delta_g}")
        if not math.isfinite(self.stderr) or self.stderr < 0:
            raise ValueError(f"stderr 必须是非负有限值：{self.stderr}")


@dataclass(frozen=True)
class EdgeResult:
    """一条边的汇总结果（计划 §4.1 的 `EdgeResult` 契约）。"""

    edge_id: str
    ligand_a_name: str
    ligand_b_name: str
    ddg_bind: float
    ddg_stderr: float
    energy_unit: str
    complex_leg: LegResult
    solvent_leg: LegResult
    covariance: float
    qualified: bool
    qualification_reasons: tuple[str, ...]
    direction: str = RBFE_DIRECTION

    def interpretation(self) -> str:
        """把符号翻译成人话——这是最容易被读反的地方。"""
        if self.ddg_bind < 0:
            better = self.ligand_b_name
        elif self.ddg_bind > 0:
            better = self.ligand_a_name
        else:
            return "两个配体的结合自由能相同"
        return f"{better} 结合更强（ΔΔG = {self.ddg_bind:+.4f} {self.energy_unit}）"

    def to_dict(self) -> dict:
        """写进 `rbfe_result.json`（计划 §7）。"""
        return {
            "edge_id": self.edge_id,
            "direction": self.direction,
            "ligand_A": self.ligand_a_name,
            "ligand_B": self.ligand_b_name,
            "ddG_bind_B_minus_A": float(self.ddg_bind),
            "ddG_stderr": float(self.ddg_stderr),
            "energy_unit": self.energy_unit,
            "interpretation": self.interpretation(),
            "legs": {
                leg.phase: {
                    "delta_g_A_to_B": float(leg.delta_g),
                    "stderr": float(leg.stderr),
                    "uncertainty_method": leg.uncertainty_method,
                    "n_effective_samples": int(leg.n_effective_samples),
                    "quality_gate_passed": bool(leg.quality_gate_passed),
                    "artifacts_fingerprint": leg.artifacts_fingerprint,
                }
                for leg in (self.complex_leg, self.solvent_leg)
            },
            "covariance": float(self.covariance),
            "qualified": bool(self.qualified),
            "qualification_reasons": list(self.qualification_reasons),
            "rbfe_core_protocol_version": RBFE_CORE_PROTOCOL_VERSION,
        }


def convert_energy(value: float, from_unit: str, to_unit: str) -> float:
    """kJ/mol <-> kcal/mol。"""
    if from_unit not in SUPPORTED_ENERGY_UNITS or to_unit not in SUPPORTED_ENERGY_UNITS:
        raise ValueError(f"不支持的单位换算：{from_unit!r} -> {to_unit!r}")
    if from_unit == to_unit:
        return float(value)
    if from_unit == KJ_PER_MOL:
        return float(value) / _KJ_PER_KCAL
    return float(value) * _KJ_PER_KCAL


def combine_rbfe(
    complex_result: LegResult,
    solvent_result: LegResult,
    *,
    covariance: float = 0.0,
) -> EdgeResult:
    """两腿 -> ΔΔG（计划 §3 的符号约定 + §7 的误差传播）。

    ::

        ΔG_complex(A→B) = G_complex(B) - G_complex(A)
        ΔG_solvent(A→B) = G_solvent(B) - G_solvent(A)

        ΔΔG_bind(B-A)  = ΔG_complex(A→B) - ΔG_solvent(A→B)

    **负的 ΔΔG 表示 B 的结合自由能更低（结合更强）。**

    误差传播（计划 §7）：ΔΔG 是两腿之差，所以

        Var(ΔΔG) = Var_complex + Var_solvent - 2·Cov(complex, solvent)

    两腿独立时 `covariance=0`，方差相加（注意：**相加，不是相减**——差的方差
    仍然是和）。若两腿共享输入构型、共享 seed 或以任何方式相关，必须显式传入
    协方差，不能当成独立。

    🔑 本函数不加 Boresch 解析释放项。计划 §3：RBFE 不机械照搬 ABFE 的释放项；
    若采用限制性 restraints，其对目标自由能的影响必须另行证明两腿抵消或计算修正，
    那是 R1/R3 的工作，不能在这里偷偷补一项。
    """
    if complex_result.phase != PHASE_COMPLEX:
        raise ValueError(f"第一个参数必须是 complex 腿：收到 phase={complex_result.phase!r}")
    if solvent_result.phase != PHASE_SOLVENT:
        raise ValueError(f"第二个参数必须是 solvent 腿：收到 phase={solvent_result.phase!r}")

    # 身份校验：两腿必须来自同一条边、同一对配体、同一方向。计划 §4.1 的
    # EdgeResult 契约要求"经过身份校验的两腿结果"——不校验就可能把两次不同运行
    # 的腿拼在一起，得到一个看着正常的假数字。
    for attr, label in (
        ("edge_id", "edge_id"),
        ("ligand_a_name", "ligand_A"),
        ("ligand_b_name", "ligand_B"),
        ("direction", "direction"),
        ("energy_unit", "energy_unit"),
    ):
        cv, sv = getattr(complex_result, attr), getattr(solvent_result, attr)
        if cv != sv:
            raise ValueError(
                f"两腿的 {label} 不一致（complex={cv!r} / solvent={sv!r}）——"
                "拒绝拼接来自不同运行的腿"
            )

    if not math.isfinite(covariance):
        raise ValueError(f"covariance 必须是有限值：{covariance}")

    ddg = complex_result.delta_g - solvent_result.delta_g

    variance = (
        complex_result.stderr**2 + solvent_result.stderr**2 - 2.0 * float(covariance)
    )
    if variance < 0.0:
        # 只可能来自一个不自洽的协方差输入。宁可报错也不 clamp 到 0——
        # clamp 会把"输入自相矛盾"伪装成"误差很小"。
        raise ValueError(
            f"传播后方差为负（{variance:.6g}）：covariance={covariance:.6g} 与两腿 "
            f"stderr（{complex_result.stderr:.6g} / {solvent_result.stderr:.6g}）不自洽"
        )

    reasons: list[str] = []
    if not complex_result.quality_gate_passed:
        reasons.append("complex 腿未通过质量门")
    if not solvent_result.quality_gate_passed:
        reasons.append("solvent 腿未通过质量门")

    return EdgeResult(
        edge_id=complex_result.edge_id,
        ligand_a_name=complex_result.ligand_a_name,
        ligand_b_name=complex_result.ligand_b_name,
        ddg_bind=ddg,
        ddg_stderr=math.sqrt(variance),
        energy_unit=complex_result.energy_unit,
        complex_leg=complex_result,
        solvent_leg=solvent_result,
        covariance=float(covariance),
        qualified=not reasons,
        qualification_reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# R1a：分子图、环分析与片段分解
# ---------------------------------------------------------------------------
#
# 计划 §5.2 对映射的硬要求（逐条）：
#   - 分子内索引、complex 全局索引、solvent 全局索引和 hybrid 索引分别记录；
#   - 验证一对一、索引范围、化学一致性和**映射核心的连通性**；
#   - 识别对称等价映射与姿势歧义；
#   - 两腿使用同一份**冻结**的分子级 A→B 映射，再各自投影为全局原子索引；
#   - 映射评分只用于候选排序，不能代替化学与几何验收。
#
# 路线：`docs/design/PROPOSAL_rbfe_r1_fragment_mapping.md` §3 的 **A+B 混合**
# （2026-09-03 用户拍板）——在非环可旋转键处切开，先做片段级匹配（可人眼审），
# 片段内部再做原子级对齐。
#
# ## 为什么这一层不 import openmm、也不 import rdkit
#
# 计划 §5.1 的红线：「**不能为了调用第三方 builder 悄悄把当前力场重参数化。**」
# 本层只吃**键图**——原子序数 + 键连接——不做任何化学感知：不推断键级、不重新
# 分配电荷、不重新参数化。用户给的那份参数从头到尾原样不动。
#
# rdkit 只在一个地方可能被用到（片段内 MCS，见 `map_atoms` 的 M4 步），且：
#   - 只在**已配对的片段内部**、只在两个片段不同构时才启用；
#   - 建 RWMol 时全部键按单键建、比较时忽略键级（`BondCompareAny`），
#     所以仍然没有引入一次化学感知；
#   - 用完即弃，不参与建系。
#
# ## 与 ABFE 那边的关系：只借设计，不 import
#
# `outer_lambda_neural_basis.py::discover_ligand_rotatable_torsions`（:5637）里有
# 一份等价的键图 + 环判定实现，本节的 `_alternate_path_exists` 与「非环可旋转键」
# 判据与它**语义一致**（同一套定义：重原子、非环、两端各至少还有一个重原子邻居）。
# 但那份实现是 `discover_ligand_rotatable_torsions` 内部的闭包，import 不出来；
# 而且它是 ABFE 侧模块，本模块按依赖方向不得 import ABFE。因此这里是**重写**，
# 不是复制粘贴，也不改 ABFE 那边任何一行。

#: 原子映射层的协议版本。进 `atom_mapping.json` 与边身份指纹。
#: 任何改变图构建、环判定、切键判据或匹配算法的改动都必须 +1——
#: 换了映射就必须拒绝复用旧产物（计划 §7）。
RBFE_MAPPING_PROTOCOL_VERSION = 1


class RBFEMappingError(RBFEValidationError):
    """原子映射无法在**确定性**下完成时抛出。

    映射歧义一律 fail closed：宁可拒绝，也不在多个等价解里随手挑一个——
    计划 §5.2 要求「识别对称等价映射与姿势歧义」，识别出来却静默选一个
    等于没识别。
    """


#: 只列本层实际会遇到的元素；缺的宁可报错也不猜。
_ELEMENT_SYMBOL_BY_Z = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 12: "Mg",
    14: "Si", 15: "P", 16: "S", 17: "Cl", 19: "K", 20: "Ca", 26: "Fe",
    30: "Zn", 35: "Br", 53: "I",
}


@dataclass(frozen=True)
class AtomNode:
    """图里的一个原子。

    `index` 是**该原子在其来源容器里的索引**（OpenMM topology 的全局索引，或
    GROMACS `.itp` 里的 1-based 序号）。映射层内部一律用它作键；分子内索引
    （0..n-1）另行记录，见 `MolecularGraph.local_index`。计划 §5.2 要求这几套
    索引**分别**记录，不能混。
    """

    index: int
    atomic_number: int
    name: str = ""
    residue_name: str = ""
    residue_index: int = 0
    chain_index: int = 0

    @property
    def element(self) -> str:
        symbol = _ELEMENT_SYMBOL_BY_Z.get(int(self.atomic_number))
        if symbol is None:
            raise RBFEMappingError(
                f"原子 index={self.index} name={self.name!r} 的原子序数 "
                f"{self.atomic_number} 不在已知元素表里。"
                "映射依赖元素身份做化学一致性检查，猜不得——请补 _ELEMENT_SYMBOL_BY_Z。"
            )
        return symbol

    @property
    def is_heavy(self) -> bool:
        return int(self.atomic_number) > 1

    def identity(self) -> dict:
        """跨 System 重建仍可审计的原子身份（计划 §5.2）。"""
        return {
            "index": int(self.index),
            "atomic_number": int(self.atomic_number),
            "element": self.element,
            "name": self.name,
            "residue_name": self.residue_name,
            "residue_index": int(self.residue_index),
            "chain_index": int(self.chain_index),
        }


@dataclass(frozen=True)
class MolecularGraph:
    """一个配体分子的键图。**只含配体原子**，不含受体和溶剂。

    构造后不可变：映射一旦基于某个图算出来，图本身就是身份的一部分。
    """

    atoms: tuple
    bonds: tuple

    def __post_init__(self) -> None:
        if not self.atoms:
            raise RBFEMappingError("分子图为空——没有原子")
        indices = [int(a.index) for a in self.atoms]
        if len(set(indices)) != len(indices):
            dupes = sorted({i for i in indices if indices.count(i) > 1})
            raise RBFEMappingError(f"分子图存在重复原子索引：{dupes}")
        index_set = set(indices)

        canonical = set()
        for pair in self.bonds:
            left, right = int(pair[0]), int(pair[1])
            if left == right:
                raise RBFEMappingError(f"自环键：atom {left} 连到自己")
            if left not in index_set or right not in index_set:
                raise RBFEMappingError(
                    f"键 ({left}, {right}) 的端点不在本图的原子集合里——"
                    "配体子图必须自洽：跨配体边界的键说明 ligand_indices 划错了"
                )
            canonical.add((min(left, right), max(left, right)))

        object.__setattr__(self, "atoms", tuple(sorted(self.atoms, key=lambda a: a.index)))
        object.__setattr__(self, "bonds", tuple(sorted(canonical)))

        neighbors = {i: set() for i in index_set}
        for left, right in self.bonds:
            neighbors[left].add(right)
            neighbors[right].add(left)
        object.__setattr__(self, "_neighbors", {i: tuple(sorted(v)) for i, v in neighbors.items()})
        object.__setattr__(self, "_by_index", {int(a.index): a for a in self.atoms})
        object.__setattr__(
            self, "_local_index", {int(a.index): n for n, a in enumerate(self.atoms)}
        )
        object.__setattr__(self, "_ring_bonds_cache", None)

        if not self._is_connected():
            raise RBFEMappingError(
                "配体子图不连通——它被拆成了多个互不相连的片段。"
                "共价配体、或者 ligand_indices 混进了别的分子，都会长这样。"
                "首版拒绝（计划 §2 的共价配体条款）。"
            )

    # -- 基本访问 ----------------------------------------------------------

    @property
    def indices(self) -> tuple:
        return tuple(int(a.index) for a in self.atoms)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    def atom(self, index: int):
        try:
            return self._by_index[int(index)]
        except KeyError:
            raise RBFEMappingError(f"原子 {index} 不在本图里") from None

    def local_index(self, index: int) -> int:
        """全局索引 → 分子内索引（0..n-1，按全局索引升序）。"""
        try:
            return self._local_index[int(index)]
        except KeyError:
            raise RBFEMappingError(f"原子 {index} 不在本图里") from None

    def neighbors(self, index: int) -> tuple:
        try:
            return self._neighbors[int(index)]
        except KeyError:
            raise RBFEMappingError(f"原子 {index} 不在本图里") from None

    def heavy_neighbors(self, index: int) -> tuple:
        return tuple(i for i in self.neighbors(index) if self.atom(i).is_heavy)

    def degree(self, index: int) -> int:
        return len(self.neighbors(index))

    def _is_connected(self) -> bool:
        start = int(self.atoms[0].index)
        seen = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in self._neighbors[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return len(seen) == len(self.atoms)

    # -- M1：环分析 --------------------------------------------------------

    def _alternate_path_exists(self, left: int, right: int) -> bool:
        """去掉 (left, right) 这条键之后，left 还能不能走到 right。

        能走到 ⇒ 这条键在环上。与 `discover_ligand_rotatable_torsions` 里的
        `central_bond_is_in_ring` 语义一致（见本节顶部说明）。
        """
        stack = [left]
        visited = {left}
        while stack:
            current = stack.pop()
            for neighbor in self._neighbors[current]:
                if {current, neighbor} == {left, right}:
                    continue
                if neighbor == right:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return False

    def ring_bonds(self) -> frozenset:
        """所有处于环上的键。"""
        if self._ring_bonds_cache is None:
            found = frozenset(
                bond for bond in self.bonds if self._alternate_path_exists(bond[0], bond[1])
            )
            object.__setattr__(self, "_ring_bonds_cache", found)
        return self._ring_bonds_cache

    def ring_atoms(self) -> frozenset:
        """所有处于环上的原子。

        一个原子在环上 ⟺ 它至少连着一条环键。（若键 (X,a) 在环上，按定义
        去掉它之后 X 仍能走到 a，即 X 与 a 同处一个圈里。）
        """
        found = set()
        for left, right in self.ring_bonds():
            found.add(left)
            found.add(right)
        return frozenset(found)

    def smallest_ring_size(self, left: int, right: int):
        """经过键 (left, right) 的最小环的大小；不是环键则返回 None。

        做法：删掉这条键后做 BFS 求最短路，环大小 = 最短路长度 + 1。
        """
        key = (min(left, right), max(left, right))
        if key not in self.ring_bonds():
            return None
        source, target = key
        distance = {source: 0}
        queue = [source]
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            for neighbor in self._neighbors[current]:
                if {current, neighbor} == {source, target}:
                    continue
                if neighbor in distance:
                    continue
                distance[neighbor] = distance[current] + 1
                if neighbor == target:
                    return distance[neighbor] + 1
                queue.append(neighbor)
        # ring_bonds() 说它在环上，BFS 却走不通 —— 只可能是内部状态被破坏了。
        raise RBFEMappingError(
            f"内部不一致：键 {key} 被判为环键但找不到替代通路"
        )  # pragma: no cover - 防御性

    def ring_size_profile(self) -> tuple:
        """全图环键的最小环大小的**排序多重集**。

        这是一个廉价、确定性的环指纹：环断裂/闭合会改变环键数量，环尺寸变化
        会改变其中的数值。R0 把这两类挂在 `unchecked` 里等的就是它。
        """
        sizes = [self.smallest_ring_size(l, r) for l, r in sorted(self.ring_bonds())]
        return tuple(sorted(s for s in sizes if s is not None))

    def element_counts(self) -> dict:
        counts = {}
        for atom in self.atoms:
            counts[atom.element] = counts.get(atom.element, 0) + 1
        return dict(sorted(counts.items()))

    # -- 构造入口 ----------------------------------------------------------

    @classmethod
    def from_atoms_and_bonds(cls, atoms, bonds) -> "MolecularGraph":
        return cls(atoms=tuple(atoms), bonds=tuple(tuple(b) for b in bonds))

    @classmethod
    def from_openmm_topology(cls, topology, ligand_indices) -> "MolecularGraph":
        """从 OpenMM topology 抽出配体子图。

        openmm 在这里**惰性**使用（只读 topology 的属性，不 import 模块），
        所以 `rbfe_core` 仍然可以在没装 openmm 的环境里 import——R0 的
        「不启动 GPU」性质不因为 R1 而丢掉。
        """
        wanted = {int(i) for i in ligand_indices}
        if len(wanted) < 2:
            raise RBFEMappingError(f"ligand_indices 至少需要两个原子：收到 {len(wanted)}")
        all_atoms = list(topology.atoms())
        out_of_range = sorted(i for i in wanted if i < 0 or i >= len(all_atoms))
        if out_of_range:
            raise RBFEMappingError(
                f"ligand_indices 超出 topology（共 {len(all_atoms)} 个原子）：{out_of_range}"
            )
        nodes = []
        for index in sorted(wanted):
            atom = all_atoms[index]
            element = getattr(atom, "element", None)
            atomic_number = getattr(element, "atomic_number", None) if element is not None else None
            if atomic_number is None:
                raise RBFEMappingError(
                    f"topology 原子 {index}（{getattr(atom, 'name', '?')}）没有 element——"
                    "映射需要元素身份，不能从原子名猜"
                )
            residue = atom.residue
            nodes.append(
                AtomNode(
                    index=int(index),
                    atomic_number=int(atomic_number),
                    name=str(getattr(atom, "name", "")),
                    residue_name=str(residue.name),
                    residue_index=int(residue.index),
                    chain_index=int(residue.chain.index),
                )
            )
        bonds = [
            (int(a.index), int(b.index))
            for a, b in topology.bonds()
            if int(a.index) in wanted and int(b.index) in wanted
        ]
        return cls.from_atoms_and_bonds(nodes, bonds)

    @classmethod
    def from_gromacs_itp(cls, path: str, moleculetype: Optional[str] = None) -> "MolecularGraph":
        """从 GROMACS `.itp` 的 `[ atoms ]` / `[ bonds ]` 建图。

        首版锁定的输入路线是**已参数化的 GROMACS／OpenMM 输入**（计划 §5.1），
        这是其中 GROMACS 那半边的入口。只读 `[ atomtypes ]`（拿 at.num）、
        `[ atoms ]`、`[ bonds ]` 三段——不解析力场参数、不做任何转换。

        元素来自 `[ atomtypes ]` 的 at.num 列。拿不到就**报错**，不按质量或
        原子名猜：猜错一个元素，整个映射的化学一致性检查就是假的。
        """
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        section = None
        atomic_number_by_type = {}
        rows = []
        bonds = []
        current_moleculetype = None
        for raw in text.splitlines():
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("#"):
                continue          # #include / #ifdef：本层不展开
            if line.startswith("["):
                section = line.strip("[] \t").lower()
                continue
            fields = line.split()
            if section == "atomtypes":
                # name at.num mass charge ptype sigma epsilon
                if len(fields) >= 2 and fields[1].isdigit():
                    atomic_number_by_type[fields[0]] = int(fields[1])
            elif section == "moleculetype":
                current_moleculetype = fields[0]
            elif section == "atoms":
                if moleculetype is not None and current_moleculetype != moleculetype:
                    continue
                if len(fields) < 5:
                    continue
                rows.append(fields)
            elif section == "bonds":
                if moleculetype is not None and current_moleculetype != moleculetype:
                    continue
                if len(fields) >= 2:
                    bonds.append((int(fields[0]), int(fields[1])))

        if not rows:
            where = f"（moleculetype={moleculetype!r}）" if moleculetype else ""
            raise RBFEMappingError(f"{path} 里没有找到 [ atoms ] 段{where}")

        nodes = []
        for fields in rows:
            index, atom_type, resnr, resname, atom_name = fields[:5]
            atomic_number = atomic_number_by_type.get(atom_type)
            if atomic_number is None:
                raise RBFEMappingError(
                    f"{path}：原子类型 {atom_type!r} 在 [ atomtypes ] 里没有 at.num，"
                    "无法确定元素。不按质量/原子名猜——猜错元素会让化学一致性检查失效。"
                )
            nodes.append(
                AtomNode(
                    index=int(index),
                    atomic_number=int(atomic_number),
                    name=str(atom_name),
                    residue_name=str(resname),
                    residue_index=int(resnr),
                    chain_index=0,
                )
            )
        return cls.from_atoms_and_bonds(nodes, bonds)


# -- M2：片段分解 ----------------------------------------------------------


def is_fragmentation_bond(graph: MolecularGraph, left: int, right: int) -> bool:
    """这条键能不能作为片段切点：**非环、重原子—重原子、两端都不是末端**。

    与 `discover_ligand_rotatable_torsions` 的「可旋转中央键」判据同一套定义。
    末端键（甲基、羟基等）不切——切了只会得到一堆单原子片段，人眼没法审。
    """
    key = (min(left, right), max(left, right))
    if key not in set(graph.bonds):
        return False
    if key in graph.ring_bonds():
        return False
    if not (graph.atom(key[0]).is_heavy and graph.atom(key[1]).is_heavy):
        return False
    outer_left = [i for i in graph.heavy_neighbors(key[0]) if i != key[1]]
    outer_right = [i for i in graph.heavy_neighbors(key[1]) if i != key[0]]
    return bool(outer_left) and bool(outer_right)


@dataclass(frozen=True)
class Fragment:
    """一个片段：切键之后的一个连通块。"""

    fragment_id: int
    atom_indices: tuple
    #: 该片段里参与切键的原子（连到别的片段去的那些），排序后的元组。
    attachment_atoms: tuple
    element_counts: dict
    ring_size_profile: tuple
    n_internal_bonds: int

    @property
    def n_atoms(self) -> int:
        return len(self.atom_indices)

    def signature(self) -> tuple:
        """片段的**结构签名**：组成 + 环 + 内部键数 + 接点数。

        签名相同不等于同构；它只用来把候选缩到很小的范围，真正的判定在
        M4 的原子级对齐。计划 §5.2：「映射评分只用于候选排序，不能代替
        化学与几何验收。」
        """
        return (
            tuple(sorted(self.element_counts.items())),
            self.ring_size_profile,
            int(self.n_internal_bonds),
            len(self.attachment_atoms),
        )


@dataclass(frozen=True)
class FragmentDecomposition:
    """M2 的产物：片段集合 + 片段之间的连接图。"""

    graph: MolecularGraph
    fragments: tuple
    cut_bonds: tuple
    #: 原子全局索引 -> fragment_id
    fragment_of_atom: dict
    #: fragment_id -> 与之相连的 (邻居 fragment_id, 本侧接点原子, 对侧接点原子) 元组
    adjacency: dict

    def fragment(self, fragment_id: int) -> Fragment:
        return self.fragments[int(fragment_id)]

    def to_dict(self) -> dict:
        return {
            "n_fragments": len(self.fragments),
            "cut_bonds": [list(b) for b in self.cut_bonds],
            "fragments": [
                {
                    "fragment_id": f.fragment_id,
                    "atom_indices": list(f.atom_indices),
                    "attachment_atoms": list(f.attachment_atoms),
                    "element_counts": f.element_counts,
                    "ring_size_profile": list(f.ring_size_profile),
                    "n_internal_bonds": f.n_internal_bonds,
                }
                for f in self.fragments
            ],
        }


def decompose_into_fragments(graph: MolecularGraph) -> FragmentDecomposition:
    """M2：在非环可旋转键处切开，得到片段集合与片段连接图。

    片段编号是**确定性**的：按片段内最小原子索引升序编号。同一个图重复分解
    必然得到同一份编号——映射要能冻结进身份指纹，非确定性的编号会让指纹漂移。
    """
    cut_bonds = tuple(
        bond for bond in graph.bonds if is_fragmentation_bond(graph, bond[0], bond[1])
    )
    cut_set = set(cut_bonds)

    remaining = {}
    for index in graph.indices:
        remaining[index] = [
            n for n in graph.neighbors(index)
            if (min(index, n), max(index, n)) not in cut_set
        ]

    unassigned = set(graph.indices)
    blocks = []
    while unassigned:
        seed = min(unassigned)
        block = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in remaining[current]:
                if neighbor not in block:
                    block.add(neighbor)
                    stack.append(neighbor)
        blocks.append(tuple(sorted(block)))
        unassigned -= block
    blocks.sort(key=lambda b: b[0])

    fragment_of_atom = {}
    for fragment_id, block in enumerate(blocks):
        for index in block:
            fragment_of_atom[index] = fragment_id

    fragments = []
    for fragment_id, block in enumerate(blocks):
        members = set(block)
        internal_bonds = [
            b for b in graph.bonds if b[0] in members and b[1] in members and b not in cut_set
        ]
        ring_sizes = tuple(
            sorted(
                s for s in (graph.smallest_ring_size(l, r) for l, r in internal_bonds)
                if s is not None
            )
        )
        counts = {}
        for index in block:
            symbol = graph.atom(index).element
            counts[symbol] = counts.get(symbol, 0) + 1
        attachments = tuple(
            sorted({b[0] if b[0] in members else b[1] for b in cut_set
                    if b[0] in members or b[1] in members})
        )
        fragments.append(
            Fragment(
                fragment_id=fragment_id,
                atom_indices=block,
                attachment_atoms=attachments,
                element_counts=dict(sorted(counts.items())),
                ring_size_profile=ring_sizes,
                n_internal_bonds=len(internal_bonds),
            )
        )

    adjacency = {f.fragment_id: [] for f in fragments}
    for left, right in cut_bonds:
        fl, fr = fragment_of_atom[left], fragment_of_atom[right]
        adjacency[fl].append((fr, left, right))
        adjacency[fr].append((fl, right, left))
    adjacency = {k: tuple(sorted(v)) for k, v in adjacency.items()}

    return FragmentDecomposition(
        graph=graph,
        fragments=tuple(fragments),
        cut_bonds=cut_bonds,
        fragment_of_atom=dict(sorted(fragment_of_atom.items())),
        adjacency=adjacency,
    )


# -- M3/M4：片段匹配与片段内原子对齐 -------------------------------------
#
# 整体流程（提案 §4 的 M0-M6）：
#
#   M0/M1/M2  建图 → 环分析 → 片段分解            （上面）
#   M3        片段级匹配：种子 + 沿片段图生长      （本节）
#   M4        已配对片段内部的原子级对齐（图同构）  （本节）
#   M3.5/M4b  位置对应但签名不同的片段 → rdkit MCS  （本节，路线 B 的那一半）
#   M5        组装 AtomMapping，冻结、落盘          （本节）
#   M6        映射验证                              （本节）
#
# 全程 fail closed：任何一步出现无法确定性消解的歧义，就抛 `RBFEMappingError`，
# 不在等价解里随手挑一个。

#: 同构搜索最多枚举多少个解。只用来**识别**对称等价（>1 即存在对称），
#: 不需要枚举完；到顶就停并如实记录。
_MAX_ISOMORPHISM_SOLUTIONS = 64


def _fragment_signature(
    decomposition: FragmentDecomposition, fragment_id: int
) -> tuple:
    """片段签名 + 它在片段图里的度。

    度必须进签名：组成完全相同但一个是末端基团、一个是中间连接子，化学处境
    不同，不该被当成同一个候选。
    """
    fragment = decomposition.fragment(fragment_id)
    return fragment.signature() + (len(decomposition.adjacency[fragment_id]),)


def _find_fragment_isomorphisms(
    graph_a: MolecularGraph,
    atoms_a: Sequence[int],
    graph_b: MolecularGraph,
    atoms_b: Sequence[int],
    anchors: Optional[dict] = None,
    limit: int = _MAX_ISOMORPHISM_SOLUTIONS,
) -> list:
    """枚举两个片段之间的**诱导子图同构**（全覆盖，一一对应）。

    约束（缺一不可）：
      - 元素相同——化学一致性，计划 §5.2；
      - **在整图里的度**相同——这条把片段的对外接点也一起对上了，不然会出现
        「片段内部长得一样、但一个接着苯环一个接着甲基」的错配；
      - 环原子身份相同；
      - 片段内部键关系保持（诱导，不只是保边）；
      - `anchors` 给定的原子对必须成立。

    返回 A 全局索引 -> B 全局索引 的字典列表，按确定性顺序。空列表 = 不同构。
    """
    atoms_a = sorted(int(i) for i in atoms_a)
    atoms_b = sorted(int(i) for i in atoms_b)
    if len(atoms_a) != len(atoms_b):
        return []

    ring_a, ring_b = graph_a.ring_atoms(), graph_b.ring_atoms()
    set_a, set_b = set(atoms_a), set(atoms_b)
    bonds_a = {
        (l, r) for l, r in graph_a.bonds if l in set_a and r in set_a
    }
    bonds_b = {
        (l, r) for l, r in graph_b.bonds if l in set_b and r in set_b
    }
    anchors = dict(anchors or {})

    def compatible(a: int, b: int) -> bool:
        if graph_a.atom(a).atomic_number != graph_b.atom(b).atomic_number:
            return False
        if graph_a.degree(a) != graph_b.degree(b):
            return False
        if (a in ring_a) != (b in ring_b):
            return False
        if a in anchors and anchors[a] != b:
            return False
        if b in anchors.values() and anchors.get(a) != b:
            return False
        return True

    # 先按"候选数最少"排序可以剪掉大量分支，但会让枚举顺序依赖数据。
    # 这里坚持**按全局索引升序**：解的顺序必须可复现，才谈得上"规范解"。
    solutions = []

    def backtrack(position: int, mapping: dict, used: set) -> None:
        if len(solutions) >= limit:
            return
        if position == len(atoms_a):
            solutions.append(dict(mapping))
            return
        a = atoms_a[position]
        for b in atoms_b:
            if b in used or not compatible(a, b):
                continue
            ok = True
            for mapped_a, mapped_b in mapping.items():
                bond_in_a = (min(a, mapped_a), max(a, mapped_a)) in bonds_a
                bond_in_b = (min(b, mapped_b), max(b, mapped_b)) in bonds_b
                if bond_in_a != bond_in_b:
                    ok = False
                    break
            if not ok:
                continue
            mapping[a] = b
            used.add(b)
            backtrack(position + 1, mapping, used)
            del mapping[a]
            used.discard(b)

    backtrack(0, {}, set())
    return solutions


def _canonical_solution(solutions: list) -> dict:
    """在多个等价解里选**规范解**：按 (A 索引升序下的 B 索引序列) 字典序最小。

    这只是"选一个可复现的"，不代表其它解是错的。对称等价解的数量会被单独
    记进映射产物，不藏起来。
    """
    return min(solutions, key=lambda s: tuple(s[k] for k in sorted(s)))


def _select_seed(
    da: FragmentDecomposition,
    db: FragmentDecomposition,
    sig_a: dict,
    sig_b: dict,
) -> tuple:
    """M3 的种子：一个在 A 和 B 里**签名都只出现一次**的片段对。

    种子必须唯一，否则从第一步就在猜。优先取原子数最多、环最多的那个——
    骨架片段最不容易撞签名，也最不容易在首版范围（相近骨架）里被改掉。

    找不到唯一种子就 fail closed：这通常意味着两个分子要么高度对称、要么差异
    已经超出「同一受体、相近骨架」的首版范围（计划 §2）。
    """

    def unique_signatures(sigs: dict) -> set:
        counts = {}
        for value in sigs.values():
            counts[value] = counts.get(value, 0) + 1
        return {k for k, v in counts.items() if v == 1}

    shared = unique_signatures(sig_a) & unique_signatures(sig_b)
    if not shared:
        raise RBFEMappingError(
            "找不到可作为种子的片段：没有任何片段签名在 A 和 B 里都只出现一次。"
            "这说明两个分子的片段分解高度对称或差异过大，首版不猜——"
            "请检查是不是超出了「同一受体、相近骨架」的首版范围（计划 §2）。"
        )

    def seed_key(signature) -> tuple:
        fragment_id = next(k for k, v in sig_a.items() if v == signature)
        fragment = da.fragment(fragment_id)
        return (-fragment.n_atoms, -len(fragment.ring_size_profile), fragment_id)

    signature = min(shared, key=seed_key)
    seed_a = next(k for k, v in sig_a.items() if v == signature)
    seed_b = next(k for k, v in sig_b.items() if v == signature)
    return seed_a, seed_b


def _anchor_constraints(
    da: FragmentDecomposition,
    db: FragmentDecomposition,
    fragment_pairs: dict,
    fa: int,
    fb: int,
    known_pairs: dict,
) -> dict:
    """已配对片段 (fa, fb) 内部对齐时的锚点：接点原子必须对上接点原子。

    锚点有两个来源，强度不同：

    1. **切键对端已定**（强）：fa 经切键连到已配对的邻居 nfa，且对端原子
       `other_a` 已经在 `known_pairs` 里——那么本侧接点 `own_a` 只能对到 B 侧
       那条"对端正好是 core[other_a]"的切键的接点。多重连接也能唯一确定。
    2. **一对一连接**（弱，兜底）：fa↔nfa 与 fb↔nfb 之间各恰好一条切键时，
       两个接点直接对上。

    都定不下来就**不加锚**，交给同构自己解；解不唯一会被如实记成对称等价。
    """
    anchors = {}
    for neighbor_a, own_a, other_a in da.adjacency[fa]:
        if neighbor_a not in fragment_pairs:
            continue
        neighbor_b = fragment_pairs[neighbor_a]
        links_b = [e for e in db.adjacency[fb] if e[0] == neighbor_b]
        mapped_other = known_pairs.get(other_a)
        if mapped_other is not None:
            exact = [e for e in links_b if e[2] == mapped_other]
            if len(exact) == 1:
                anchors[own_a] = exact[0][1]
                continue
        links_a = [e for e in da.adjacency[fa] if e[0] == neighbor_a]
        if len(links_a) == 1 and len(links_b) == 1:
            anchors[own_a] = links_b[0][1]
    # 已经在别处确定下来的原子对（例如上一轮 MCS 定下的）同样是锚
    for atom_a in da.fragment(fa).atom_indices:
        if atom_a in known_pairs:
            anchors[atom_a] = known_pairs[atom_a]
    return anchors


def _build_rdkit_fragment(graph: MolecularGraph, atom_indices: Sequence[int]):
    """把一个片段建成 rdkit RWMol：**全部单键、不做化学感知**。

    这是路线 A+B 里 rdkit 唯一出场的地方。刻意不推断键级、不 sanitize 芳香性：
    计划 §5.1 不允许为了调第三方库把输入重新解释一遍。MCS 那边配套用
    `BondCompare.CompareAny` 忽略键级，所以丢掉键级不影响匹配。
    """
    from rdkit import Chem  # 惰性 import：没装 rdkit 也能 import rbfe_core

    order = sorted(int(i) for i in atom_indices)
    position = {index: n for n, index in enumerate(order)}
    mol = Chem.RWMol()
    for index in order:
        atom = Chem.Atom(int(graph.atom(index).atomic_number))
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    members = set(order)
    for left, right in graph.bonds:
        if left in members and right in members:
            mol.AddBond(position[left], position[right], Chem.BondType.SINGLE)
    result = mol.GetMol()
    result.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(result)
    return result, order


def _mcs_align(
    graph_a: MolecularGraph,
    atoms_a: Sequence[int],
    graph_b: MolecularGraph,
    atoms_b: Sequence[int],
    anchors: dict,
) -> dict:
    """M4b：两个**签名不同但位置对应**的片段之间求最大公共子结构。

    用途：A→B 的差异基团往往落在某一个片段里。没有这一步，整个片段会整块变成
    dummy——正确但浪费（公共核心变小、收敛更差）。有了它，只有真正变掉的那几个
    原子才是 dummy。

    rdkit 缺失时返回空字典，由调用方降级为"整块 dummy"并**在 method 里记下来**
    （降级会改变映射，必须可见，不能静默）。
    """
    try:
        from rdkit.Chem import rdFMCS
    except ImportError:
        return {}

    mol_a, order_a = _build_rdkit_fragment(graph_a, atoms_a)
    mol_b, order_b = _build_rdkit_fragment(graph_b, atoms_b)
    result = rdFMCS.FindMCS(
        [mol_a, mol_b],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,      # 不允许 MCS 只吃半个环——那正是"环断裂"
        matchValences=False,
        timeout=30,
    )
    if result.canceled or result.numAtoms == 0:
        return {}

    from rdkit import Chem

    query = Chem.MolFromSmarts(result.smartsString)
    if query is None:
        return {}
    matches_a = mol_a.GetSubstructMatches(query, uniquify=False, maxMatches=256)
    matches_b = mol_b.GetSubstructMatches(query, uniquify=False, maxMatches=256)
    if not matches_a or not matches_b:
        return {}

    best = None
    for match_a in matches_a:
        for match_b in matches_b:
            candidate = {
                order_a[pa]: order_b[pb] for pa, pb in zip(match_a, match_b)
            }
            if any(
                candidate.get(k) not in (None, v) for k, v in anchors.items()
            ):
                continue
            if any(
                graph_a.atom(k).atomic_number != graph_b.atom(v).atomic_number
                for k, v in candidate.items()
            ):
                continue
            key = tuple(candidate[k] for k in sorted(candidate))
            if best is None or key < best[0]:
                best = (key, candidate)
    return best[1] if best is not None else {}


@dataclass(frozen=True)
class AtomMapping:
    """冻结的分子级 A→B 映射（计划 §5.2）。

    **索引一律是分子内索引**（0..n-1，按来源索引升序）。计划 §5.2 要求两腿共用
    同一份分子级映射、再各自投影为全局索引——所以这个对象里不存 complex/solvent
    的全局索引，投影由 `project()` 在每条腿上单独做。这样物理上不可能出现
    "两腿用了不同映射"。
    """

    protocol_version: int
    n_atoms_a: int
    n_atoms_b: int
    #: ((local_a, local_b), ...)，按 local_a 升序。公共核心。
    core_pairs: tuple
    #: 只存在于 A 的原子（分子内索引）——B 侧为 dummy。
    a_only: tuple
    #: 只存在于 B 的原子——A 侧为 dummy。
    b_only: tuple
    #: ((fragment_id_a, fragment_id_b), ...)
    fragment_pairs: tuple
    method: str
    #: ((fragment_id_a, fragment_id_b, n_solutions), ...)，n>1 即存在对称等价解
    symmetry_solution_counts: tuple
    ambiguities: tuple
    atom_identity_a: tuple
    atom_identity_b: tuple

    @property
    def n_core(self) -> int:
        return len(self.core_pairs)

    def a_to_b(self) -> dict:
        return {int(a): int(b) for a, b in self.core_pairs}

    def b_to_a(self) -> dict:
        return {int(b): int(a) for a, b in self.core_pairs}

    def hybrid_indices(self) -> dict:
        """hybrid 体系里的原子编号：core → A-only → B-only，各段内保序。

        返回 {"a": {local_a: hybrid}, "b": {local_b: hybrid}, "n_hybrid_atoms": n}。
        计划 §5.3 要求"保持跨 λ 相同的粒子、质量和约束结构"——hybrid 体系的粒子
        集合因此是三段的并集，两个端点各自只是其中的一个子集。
        """
        map_a, map_b = {}, {}
        cursor = 0
        for local_a, local_b in self.core_pairs:
            map_a[int(local_a)] = cursor
            map_b[int(local_b)] = cursor
            cursor += 1
        for local_a in self.a_only:
            map_a[int(local_a)] = cursor
            cursor += 1
        for local_b in self.b_only:
            map_b[int(local_b)] = cursor
            cursor += 1
        return {"a": map_a, "b": map_b, "n_hybrid_atoms": cursor}

    def project(self, global_indices_a: Sequence[int], global_indices_b: Sequence[int]) -> dict:
        """把分子级映射投影到某一条腿的全局原子索引上（计划 §5.2）。

        `global_indices_*` 是该配体在这条腿的体系里的全局索引，**按分子内顺序**
        给出（即第 i 个元素就是分子内索引 i 的那个原子）。长度对不上直接报错——
        投影错位是那种能一路跑完、结果全错的失败模式。
        """
        list_a = [int(i) for i in global_indices_a]
        list_b = [int(i) for i in global_indices_b]
        if len(list_a) != self.n_atoms_a:
            raise RBFEMappingError(
                f"投影失败：给了 {len(list_a)} 个 A 的全局索引，映射里 A 有 "
                f"{self.n_atoms_a} 个原子"
            )
        if len(list_b) != self.n_atoms_b:
            raise RBFEMappingError(
                f"投影失败：给了 {len(list_b)} 个 B 的全局索引，映射里 B 有 "
                f"{self.n_atoms_b} 个原子"
            )
        for label, values in (("A", list_a), ("B", list_b)):
            if len(set(values)) != len(values):
                raise RBFEMappingError(f"投影失败：{label} 的全局索引有重复")
        return {
            "core_pairs": tuple((list_a[a], list_b[b]) for a, b in self.core_pairs),
            "a_only": tuple(list_a[i] for i in self.a_only),
            "b_only": tuple(list_b[i] for i in self.b_only),
        }

    def to_dict(self) -> dict:
        """写进 `atom_mapping.json` 的内容（计划 §7）。"""
        return {
            "rbfe_mapping_protocol_version": int(self.protocol_version),
            "method": self.method,
            "n_atoms_A": int(self.n_atoms_a),
            "n_atoms_B": int(self.n_atoms_b),
            "n_core": self.n_core,
            "core_pairs_molecule_local": [[int(a), int(b)] for a, b in self.core_pairs],
            "A_only_molecule_local": [int(i) for i in self.a_only],
            "B_only_molecule_local": [int(i) for i in self.b_only],
            "fragment_pairs": [[int(a), int(b)] for a, b in self.fragment_pairs],
            "symmetry_solution_counts": [
                [int(a), int(b), int(n)] for a, b, n in self.symmetry_solution_counts
            ],
            "ambiguities": list(self.ambiguities),
            "hybrid_indices": {
                "A": {str(k): v for k, v in self.hybrid_indices()["a"].items()},
                "B": {str(k): v for k, v in self.hybrid_indices()["b"].items()},
                "n_hybrid_atoms": self.hybrid_indices()["n_hybrid_atoms"],
            },
            "atom_identity_A": list(self.atom_identity_a),
            "atom_identity_B": list(self.atom_identity_b),
        }

    def fingerprint(self) -> str:
        """映射的身份哈希，供 `rbfe_pipeline.edge_identity(atom_mapping_hash=...)`。

        换了映射 ⇒ 哈希变 ⇒ 旧产物不可复用（计划 §7 的硬要求）。
        故意**不含** `atom_identity_*`：那是给人看的审计信息，改个原子名不该
        让已经跑完的采样作废。
        """
        payload = {
            "v": int(self.protocol_version),
            "method": self.method,
            "core": [[int(a), int(b)] for a, b in self.core_pairs],
            "a_only": [int(i) for i in self.a_only],
            "b_only": [int(i) for i in self.b_only],
            "n_a": int(self.n_atoms_a),
            "n_b": int(self.n_atoms_b),
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def map_atoms(
    graph_a: MolecularGraph,
    graph_b: MolecularGraph,
    *,
    allow_mcs: bool = True,
) -> AtomMapping:
    """M3-M5：从两个配体的键图算出冻结的 A→B 原子映射。

    路线 A+B（提案 §3）：片段级匹配定骨架对应，片段内部做原子级对齐；只有
    「位置对应但组成不同」的那一对片段才动用 rdkit MCS。

    ## 生长为什么按接点原子走，而不是按片段签名走

    第一版按签名生长，在真实 Atenolol 上就错了：把苯环换掉一个取代基之后，
    苯环片段的签名变了 → 生长在苯环处断掉 → **苯环后面那一整段（酰胺尾）
    明明 A、B 一模一样，却被整段判成 A-only/B-only**。

    现在的规则是：先把当前片段对齐（原子级），于是它的接点原子已经进 core；
    再沿切键找邻居——**B 侧候选限定在"挂在对应接点原子上"的那些未配对片段**。
    接点对应是比签名强得多的约束，而且它不因为取代基变化而失效，所以生长能
    穿过差异片段继续往后走。签名只在同一接点上有多个候选时用来消歧。

    `allow_mcs=False` 时完全不碰 rdkit，差异片段整块进 dummy——结果仍然正确，
    只是公共核心更小。这个降级会写进 `method`，不静默。
    """
    da = decompose_into_fragments(graph_a)
    db = decompose_into_fragments(graph_b)
    sig_a = {f.fragment_id: _fragment_signature(da, f.fragment_id) for f in da.fragments}
    sig_b = {f.fragment_id: _fragment_signature(db, f.fragment_id) for f in db.fragments}

    seed_a, seed_b = _select_seed(da, db, sig_a, sig_b)

    pairs = {seed_a: seed_b}
    used_b = {seed_b}
    core: dict = {}
    symmetry_counts = []
    ambiguities = []
    mcs_used = False
    mcs_skipped = False

    queue = [(seed_a, seed_b)]
    head = 0
    while head < len(queue):
        fa, fb = queue[head]
        head += 1

        # --- M4：片段内原子级对齐 ---
        anchors = _anchor_constraints(da, db, pairs, fa, fb, core)
        solutions = _find_fragment_isomorphisms(
            graph_a,
            da.fragment(fa).atom_indices,
            graph_b,
            db.fragment(fb).atom_indices,
            anchors=anchors,
        )
        if solutions:
            chosen = _canonical_solution(solutions)
            if len(solutions) > 1:
                symmetry_counts.append((fa, fb, len(solutions)))
            core.update(chosen)
        else:
            # 不同构（组成不同，或组成相同但连接方式不同）→ M4b：MCS 求最大公共子结构
            partial = {}
            if allow_mcs:
                partial = _mcs_align(
                    graph_a,
                    da.fragment(fa).atom_indices,
                    graph_b,
                    db.fragment(fb).atom_indices,
                    anchors,
                )
            if partial:
                mcs_used = True
                core.update(partial)
            else:
                mcs_skipped = True
                ambiguities.append(
                    f"片段 A#{fa}↔B#{fb} 不同构"
                    + ("且 MCS 未能给出公共子结构" if allow_mcs else "，MCS 被关闭")
                    + "——整块进 dummy"
                )

        # --- M3：沿切键生长，按接点原子对应找 B 侧候选 ---
        for neighbor_a, own_a, _ in sorted(da.adjacency[fa]):
            if neighbor_a in pairs:
                continue
            own_b = core.get(own_a)
            candidates = [
                (nb, ob) for nb, ob, _ in db.adjacency[fb]
                if nb not in used_b and (own_b is None or ob == own_b)
            ]
            candidates = sorted({nb for nb, _ in candidates})
            if len(candidates) > 1:
                narrowed = [nb for nb in candidates if sig_b[nb] == sig_a[neighbor_a]]
                if len(narrowed) == 1:
                    candidates = narrowed
                else:
                    ambiguities.append(
                        f"片段 A#{neighbor_a}（挂在 A#{fa} 的原子 {own_a} 上）在 B 侧有 "
                        f"{len(candidates)} 个候选，签名也消不掉歧义——不猜，留作未配对"
                    )
                    continue
            if not candidates:
                continue
            neighbor_b = candidates[0]
            pairs[neighbor_a] = neighbor_b
            used_b.add(neighbor_b)
            queue.append((neighbor_a, neighbor_b))

    # --- M5：组装 ---
    local_a = {i: graph_a.local_index(i) for i in graph_a.indices}
    local_b = {i: graph_b.local_index(i) for i in graph_b.indices}
    core_pairs = tuple(sorted((local_a[a], local_b[b]) for a, b in core.items()))
    mapped_a = {a for a, _ in core_pairs}
    mapped_b = {b for _, b in core_pairs}

    unpaired_a = sorted(set(sig_a) - set(pairs))
    unpaired_b = sorted(set(sig_b) - used_b)
    for fragment_id in unpaired_a:
        ambiguities.append(f"片段 A#{fragment_id} 未能配对——整块进 A-only（dummy）")
    for fragment_id in unpaired_b:
        ambiguities.append(f"片段 B#{fragment_id} 未能配对——整块进 B-only（dummy）")

    if mcs_used and not mcs_skipped:
        method = "fragment_isomorphism+rdkit_mcs"
    elif mcs_used and mcs_skipped:
        method = "fragment_isomorphism+rdkit_mcs__partial_conservative"
    elif mcs_skipped and allow_mcs:
        method = "fragment_isomorphism_only__mcs_unavailable_conservative"
    elif mcs_skipped:
        method = "fragment_isomorphism_only__mcs_disabled_conservative"
    else:
        method = "fragment_isomorphism"

    return AtomMapping(
        protocol_version=RBFE_MAPPING_PROTOCOL_VERSION,
        n_atoms_a=graph_a.n_atoms,
        n_atoms_b=graph_b.n_atoms,
        core_pairs=core_pairs,
        a_only=tuple(sorted(set(local_a.values()) - mapped_a)),
        b_only=tuple(sorted(set(local_b.values()) - mapped_b)),
        fragment_pairs=tuple(sorted(pairs.items())),
        method=method,
        symmetry_solution_counts=tuple(symmetry_counts),
        ambiguities=tuple(ambiguities),
        atom_identity_a=tuple(a.identity() for a in graph_a.atoms),
        atom_identity_b=tuple(b.identity() for b in graph_b.atoms),
    )


# -- M6：映射验证 ----------------------------------------------------------


def validate_mapping(
    mapping: AtomMapping,
    graph_a: MolecularGraph,
    graph_b: MolecularGraph,
    *,
    edge_id: str = "atom_mapping",
) -> ValidationReport:
    """M6：一一对应、索引范围、化学一致性、核心连通性、环与元素变化。

    这一步把 R0 挂在 `unchecked` 里的几条**真正查掉**：环断裂/闭合、环尺寸变化、
    映射元素改变、共价配体（后者在建图时就被连通性挡掉了）。
    """
    report = ValidationReport(edge_id=edge_id)
    a_to_b = {}
    b_to_a = {}

    for local_a, local_b in mapping.core_pairs:
        if local_a in a_to_b:
            report.errors.append(f"映射不是一一对应：A 的原子 {local_a} 出现多次")
        if local_b in b_to_a:
            report.errors.append(f"映射不是一一对应：B 的原子 {local_b} 出现多次")
        a_to_b[local_a] = local_b
        b_to_a[local_b] = local_a

    for label, values, limit in (
        ("A", list(a_to_b) + list(mapping.a_only), mapping.n_atoms_a),
        ("B", list(b_to_a) + list(mapping.b_only), mapping.n_atoms_b),
    ):
        bad = sorted(i for i in values if i < 0 or i >= limit)
        if bad:
            report.errors.append(f"{label} 侧索引越界（合法范围 0..{limit - 1}）：{bad}")

    covered_a = set(a_to_b) | set(mapping.a_only)
    covered_b = set(b_to_a) | set(mapping.b_only)
    if covered_a != set(range(mapping.n_atoms_a)):
        missing = sorted(set(range(mapping.n_atoms_a)) - covered_a)
        report.errors.append(
            f"A 的原子没有被 core/A-only 完全划分，漏了 {missing}——"
            "hybrid 体系会少粒子"
        )
    if covered_b != set(range(mapping.n_atoms_b)):
        missing = sorted(set(range(mapping.n_atoms_b)) - covered_b)
        report.errors.append(f"B 的原子没有被 core/B-only 完全划分，漏了 {missing}")

    if not mapping.core_pairs:
        report.errors.append(
            "公共核心为空——A 与 B 之间没有任何对应原子。"
            "这不是一条 RBFE 边（那等于把 A 整个消掉再把 B 整个长出来）。"
        )

    index_a = {graph_a.local_index(i): i for i in graph_a.indices}
    index_b = {graph_b.local_index(i): i for i in graph_b.indices}

    # 化学一致性：元素必须相同（计划 §2 的"映射元素改变"）
    element_changes = []
    for local_a, local_b in mapping.core_pairs:
        if local_a not in index_a or local_b not in index_b:
            continue
        atom_a, atom_b = graph_a.atom(index_a[local_a]), graph_b.atom(index_b[local_b])
        if atom_a.atomic_number != atom_b.atomic_number:
            element_changes.append(
                f"{atom_a.name}({atom_a.element}) -> {atom_b.name}({atom_b.element})"
            )
    if element_changes:
        report.errors.append(
            "映射元素改变（首版拒绝，计划 §2）：" + "、".join(element_changes[:8])
            + ("…" if len(element_changes) > 8 else "")
        )

    # 核心连通性（计划 §5.2 明写）
    for label, graph, locals_, index_map in (
        ("A", graph_a, set(a_to_b), index_a),
        ("B", graph_b, set(b_to_a), index_b),
    ):
        members = {index_map[i] for i in locals_ if i in index_map}
        if len(members) <= 1:
            continue
        seed = min(members)
        seen = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in graph.neighbors(current):
                if neighbor in members and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        if seen != members:
            hint = ""
            if "conservative" in mapping.method:
                # 实测会踩：差异基团若在分子**中部**（例如苯环上换取代基），保守路径
                # 把整个片段丢进 dummy，核心就被这个片段从中间切断了。此时保守路径
                # 根本给不出可用映射——必须让 MCS 在片段内部保住公共骨架。
                hint = (
                    "。注意 method 是保守降级（{}）：差异片段被整块丢进 dummy。"
                    "差异基团在分子中部时这必然切断核心——请启用 rdkit MCS "
                    "（去掉 --no-mcs / allow_mcs=True），保守路径只在差异位于末端时可用"
                ).format(mapping.method)
            report.errors.append(
                f"{label} 侧的公共核心不连通（{len(seen)}/{len(members)} 个原子在同一块）——"
                f"断开的核心会让 dummy 处理无法证明抵消，首版拒绝{hint}"
            )

    # 环：断裂/闭合与尺寸变化（R0 的 unchecked 在这里被真正查掉）
    ring_a, ring_b = graph_a.ring_atoms(), graph_b.ring_atoms()
    ring_status_changes = []
    for local_a, local_b in mapping.core_pairs:
        if local_a not in index_a or local_b not in index_b:
            continue
        in_ring_a = index_a[local_a] in ring_a
        in_ring_b = index_b[local_b] in ring_b
        if in_ring_a != in_ring_b:
            ring_status_changes.append(
                f"{graph_a.atom(index_a[local_a]).name}"
                f"({'环内' if in_ring_a else '环外'} -> {'环内' if in_ring_b else '环外'})"
            )
    if ring_status_changes:
        report.errors.append(
            "环断裂/闭合：映射把环内原子对到了环外原子（首版拒绝，计划 §2）："
            + "、".join(ring_status_changes[:8])
        )

    profile_a, profile_b = graph_a.ring_size_profile(), graph_b.ring_size_profile()
    if profile_a != profile_b:
        report.warnings.append(
            f"A 与 B 的环指纹不同（{profile_a} vs {profile_b}）。"
            "若差异发生在公共核心上，上面的环断裂判据会报错；"
            "若只发生在 A-only/B-only 部分，那是被整块消掉/长出来的环，需人工确认。"
        )

    for fa, fb, count in mapping.symmetry_solution_counts:
        # 枚举到上限就停了，所以到顶时报的是**下界**，不能写成确切数字。
        how_many = (
            f"≥{count} 个（已达枚举上限 {_MAX_ISOMORPHISM_SOLUTIONS}）"
            if count >= _MAX_ISOMORPHISM_SOLUTIONS
            else f"{count} 个"
        )
        report.warnings.append(
            f"片段 A#{fa}↔B#{fb} 存在 {how_many}对称等价映射，已取规范解"
            "（按 A 索引升序下的 B 索引序列字典序最小）"
        )
    for note in mapping.ambiguities:
        report.warnings.append(note)

    if "conservative" in mapping.method:
        report.warnings.append(
            f"映射走了保守降级路径（method={mapping.method}）：差异片段整块进 dummy，"
            "公共核心比可达到的更小。结果正确但收敛更差。"
        )

    report.unchecked.extend(
        [
            "手性反转（需要坐标/立体化学感知，本层只有键图，R1 后续）",
            "结合姿势歧义（需要几何比较，R1 后续）",
            "互变异构状态改变（需要质子位置比较，R1 后续）",
            "虚拟位点/约束/自定义 Force 是否在已验证 builder 范围内（需要建系，R1b）",
        ]
    )
    return report


# ---------------------------------------------------------------------------
# R1b：受限 hybrid builder
# ---------------------------------------------------------------------------
#
# 计划 §5.3 的要求，逐条：
#   - 明确 common core / A-only / B-only，以及各自的 charge、LJ、bonded、
#     exception/1-4、约束和 dummy 处理；
#   - 路径至少描述 charge、sterics 和 bonded 项；
#   - **核对每个 λ 的有效总电荷**；
#   - 验证 λ 端点的物理相互作用恢复；
#   - 保持跨 λ 相同的粒子、质量和约束结构。
#
# ## 与 ABFE 引擎的关系（2026-09-03 用户决定）
#
# 用户明确：**直接 import `abfe_core` 复用，但不改 ABFE 一行代码。**
# 所以本节 import 是**只读**的，且一律**惰性**（在函数体内 import）——
# `abfe_core` 会拉进 openmm，放模块顶部会让 R0/R1a 那两层「不 import openmm、
# 不启动 GPU」的性质连同它们的全部测试一起失效。
#
# 复用清单（都在 ABFE 那边久经生产验证，重写必错）：
#   - `abfe_core.create_ligand_internal_force`：分子内非键 + 1-4 恢复，排除表
#     全覆盖 1-2/1-3/1-4 + **系统级约束** + exception 表
#   - `abfe_core.BeutlerSoftcoreBuilder`：softcore 表达式与 cutoff/switching 约定
#   - `ibs_engine._collect_softcore_exclusions`：跨组排除对收集
#
# ## 从 ABFE 抄 exception offset 时必须改掉的那一处
#
# `ibs_engine.py:3583` 那段的判据是 `(p1 in alchemical) ^ (p2 in alchemical)`——
# **异或**，即只处理「单端炼金」的 exception，配套还有
# `_assert_frozen_ligand_ligand_exceptions` 去断言这个前提。系数写的是
# `base=0, scale=chargeProd`，也就是插到**零**（ABFE 的端点）。
#
# RBFE 的端点不是零，是 B。所以：
#   - `addParticleParameterOffset` / `addExceptionParameterOffset` 这个**机制**
#     照用（它保 PME，是这条路上唯一正确的做法）；
#   - **系数必须换成** `base = 值ᴬ, scale = 值ᴮ − 值ᴬ`；
#   - 那条「单端」前提在 RBFE 里不成立（core–core exception 两端都在变），
#     所以 `_assert_frozen_ligand_ligand_exceptions` 不能照搬。
#
# 顺带澄清一个容易搞错的点：exception 的 `chargeProd` 是**独立参数**，不读粒子
# 电荷。所以「chargeProd 随 λ 线性插值」与「q1(λ)·q2(λ) 是 λ 的二次式」并不矛盾——
# 前者只是选了另一条路径，两个端点仍然精确。自由能只取决于端点，路径可以自选，
# 前提是平滑且无奇点。真正会出错的是**照抄系数**（那会插到零而不是插到 B）。

#: hybrid builder 的协议版本。进 hybrid System 的身份指纹。
#: 任何改变力的构造、λ 语义或分组规则的改动都必须 +1。
RBFE_HYBRID_PROTOCOL_VERSION = 1

#: 三条 λ 的全局参数名。**故意跟 ABFE 的 `lambda_coul` / `lambda_vdw` 不同名**：
#: 同名会让一个体系里两套炼金语义悄悄共享同一个全局参数。
LAMBDA_CHARGE = "lambda_rbfe_charge"
LAMBDA_STERICS = "lambda_rbfe_sterics"
LAMBDA_BONDED = "lambda_rbfe_bonded"
RBFE_LAMBDA_NAMES = (LAMBDA_CHARGE, LAMBDA_STERICS, LAMBDA_BONDED)


class RBFEHybridBuildError(RBFEValidationError):
    """hybrid 体系无法在**可证明正确**的前提下构建时抛出。

    首版受限 builder 的原则：碰到没验证过的输入就拒绝，不"尽力而为"。
    计划 §5.3 最后一句——「无法证明约束／dummy 贡献正确的变换不放行」。
    """


def _all_forces_of_type(system, force_type) -> list:
    """取出**全部**该类型的力，不是第一个。

    🔑 [P0-13] OpenMM 的 System 里同一类型的力可以有多个（GROMACS 拓扑经常
    产生两个 HarmonicBondForce）。ABFE 那边踩过一次 `next()` 只取第一个、
    配体键角被静默丢掉、分子变软的坑。这里一次都不许再踩。
    """
    return [f for f in system.getForces() if type(f) is force_type]


@dataclass(frozen=True)
class HybridTopologyLayout:
    """hybrid 体系的原子布局（计划 §5.2 的第四套索引）。

    索引约定：**环境与配体 A 的原子保持它们在 `system_a` 里的全局索引不变**，
    B-only 原子追加到末尾。这样 A 的坐标、A 的所有力项都不需要重映射，
    出错面小；代价只是 B-only 原子在编号上不与 B 的原始顺序一致（无所谓，
    映射对象本来就是身份来源）。
    """

    n_particles: int
    #: 配体 A 的分子内索引 -> hybrid 全局索引
    a_local_to_hybrid: dict
    #: 配体 B 的分子内索引 -> hybrid 全局索引
    b_local_to_hybrid: dict
    core: tuple
    a_only: tuple
    b_only: tuple
    environment: tuple

    @property
    def alchemical(self) -> tuple:
        return tuple(sorted(set(self.core) | set(self.a_only) | set(self.b_only)))

    def to_dict(self) -> dict:
        return {
            "n_particles": int(self.n_particles),
            "n_core": len(self.core),
            "n_A_only": len(self.a_only),
            "n_B_only": len(self.b_only),
            "n_environment": len(self.environment),
            "core_hybrid_indices": list(self.core),
            "A_only_hybrid_indices": list(self.a_only),
            "B_only_hybrid_indices": list(self.b_only),
        }


def _nb_force_of(system, label: str):
    """取唯一的 NonbondedForce。多于一个就拒绝——首版不处理拆分静电的体系。"""
    import openmm

    forces = _all_forces_of_type(system, openmm.NonbondedForce)
    if len(forces) != 1:
        raise RBFEHybridBuildError(
            f"{label} 有 {len(forces)} 个 NonbondedForce，首版只支持恰好 1 个。"
            "（0 个说明不是可用的力场体系；多个说明静电被拆过，"
            "本 builder 没验证过那种输入。）"
        )
    return forces[0]


def _assert_supported_forces(system, label: str) -> None:
    """首版受限 builder 只认这几种力，其余一律拒绝（计划 §2 最后一条）。"""
    import openmm

    supported = (
        openmm.HarmonicBondForce,
        openmm.HarmonicAngleForce,
        openmm.PeriodicTorsionForce,
        openmm.NonbondedForce,
        openmm.CMMotionRemover,
        openmm.MonteCarloBarostat,
    )
    unsupported = sorted(
        {type(f).__name__ for f in system.getForces() if not isinstance(f, supported)}
    )
    if unsupported:
        raise RBFEHybridBuildError(
            f"{label} 含首版 hybrid builder 未验证的力：{unsupported}。"
            "计划 §2：力场或自定义 Force 不在已验证 builder 支持范围内的输入一律拒绝。"
            "（这是本项目首版的范围限制，不是 RBFE 方法不支持。）"
        )
    for i in range(system.getNumParticles()):
        if system.isVirtualSite(i):
            raise RBFEHybridBuildError(
                f"{label} 的粒子 {i} 是虚拟位点，首版 hybrid builder 未验证（计划 §2）"
            )


def _quantity_value(value):
    """把 openmm Quantity 转成裸浮点（按其自身单位系统的默认单位）。"""
    from openmm import unit

    if hasattr(value, "value_in_unit_system"):
        return value.value_in_unit_system(unit.md_unit_system)
    return float(value)


def _assert_environment_matches(
    system_a, ligand_indices_a, system_b, ligand_indices_b
) -> tuple:
    """A、B 两个体系的**环境必须逐位相同**，否则两腿的 ΔG 差里混进了环境差异。

    比对：粒子数、质量、非键参数、环境内部的成键项、以及约束。
    返回 (env_a, env_b)：两侧环境原子索引，按顺序一一对应。

    为什么这么严：RBFE 的全部意义是「同一个环境里 A 换成 B」。环境只要有一位
    不同，ΔΔG 里就掺了一个不属于这条边的量，而且它在两腿里**不抵消**。
    """
    from openmm import unit

    set_a = {int(i) for i in ligand_indices_a}
    set_b = {int(i) for i in ligand_indices_b}
    env_a = [i for i in range(system_a.getNumParticles()) if i not in set_a]
    env_b = [i for i in range(system_b.getNumParticles()) if i not in set_b]
    if len(env_a) != len(env_b):
        raise RBFEHybridBuildError(
            f"两个体系的环境原子数不同：A 有 {len(env_a)}、B 有 {len(env_b)}。"
            "RBFE 要求同一个环境里 A 换成 B；环境不同的话 ΔΔG 里会混进环境差异，"
            "而且它在两腿里不抵消。"
        )

    for k, (ia, ib) in enumerate(zip(env_a, env_b)):
        ma = _quantity_value(system_a.getParticleMass(ia))
        mb = _quantity_value(system_b.getParticleMass(ib))
        if abs(ma - mb) > 1e-9:
            raise RBFEHybridBuildError(
                f"环境第 {k} 个原子（A#{ia} / B#{ib}）质量不同：{ma} vs {mb}"
            )

    nb_a, nb_b = _nb_force_of(system_a, "配体 A 体系"), _nb_force_of(system_b, "配体 B 体系")
    for k, (ia, ib) in enumerate(zip(env_a, env_b)):
        pa = [_quantity_value(v) for v in nb_a.getParticleParameters(ia)]
        pb = [_quantity_value(v) for v in nb_b.getParticleParameters(ib)]
        if max(abs(x - y) for x, y in zip(pa, pb)) > 1e-9:
            raise RBFEHybridBuildError(
                f"环境第 {k} 个原子（A#{ia} / B#{ib}）非键参数不同：{pa} vs {pb}"
            )

    map_a = {index: k for k, index in enumerate(env_a)}
    map_b = {index: k for k, index in enumerate(env_b)}

    def env_bonded_signature(system, env_map, ligand_set) -> tuple:
        import openmm

        rows = []
        for force in _all_forces_of_type(system, openmm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                p1, p2, length, k = force.getBondParameters(i)
                if p1 in ligand_set or p2 in ligand_set:
                    continue
                key = tuple(sorted((env_map[p1], env_map[p2])))
                rows.append(("bond", key, round(_quantity_value(length), 12), round(_quantity_value(k), 9)))
        for force in _all_forces_of_type(system, openmm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                p1, p2, p3, theta, k = force.getAngleParameters(i)
                if {p1, p2, p3} & ligand_set:
                    continue
                ends = tuple(sorted((env_map[p1], env_map[p3])))
                rows.append(("angle", (ends, env_map[p2]), round(_quantity_value(theta), 12), round(_quantity_value(k), 9)))
        for force in _all_forces_of_type(system, openmm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, n, phase, k = force.getTorsionParameters(i)
                if {p1, p2, p3, p4} & ligand_set:
                    continue
                rows.append(
                    ("torsion", tuple(env_map[p] for p in (p1, p2, p3, p4)), int(n),
                     round(_quantity_value(phase), 12), round(_quantity_value(k), 9))
                )
        return tuple(sorted(rows, key=repr))

    sig_a = env_bonded_signature(system_a, map_a, set_a)
    sig_b = env_bonded_signature(system_b, map_b, set_b)
    if sig_a != sig_b:
        only_a = [r for r in sig_a if r not in set(sig_b)]
        only_b = [r for r in sig_b if r not in set(sig_a)]
        raise RBFEHybridBuildError(
            f"两个体系的环境成键项不同：只在 A 里的 {len(only_a)} 项、"
            f"只在 B 里的 {len(only_b)} 项。示例 A={only_a[:2]} B={only_b[:2]}"
        )

    def env_constraints(system, env_map, ligand_set) -> tuple:
        rows = []
        for i in range(system.getNumConstraints()):
            p1, p2, distance = system.getConstraintParameters(i)
            if p1 in ligand_set or p2 in ligand_set:
                continue
            rows.append((tuple(sorted((env_map[p1], env_map[p2]))), round(_quantity_value(distance), 12)))
        return tuple(sorted(rows))

    if env_constraints(system_a, map_a, set_a) != env_constraints(system_b, map_b, set_b):
        raise RBFEHybridBuildError("两个体系的环境约束不同")

    return tuple(env_a), tuple(env_b)


def build_hybrid_layout(
    system_a,
    ligand_indices_a,
    system_b,
    ligand_indices_b,
    mapping: AtomMapping,
) -> HybridTopologyLayout:
    """确定 hybrid 体系的原子布局，并把所有前置条件查干净。

    这一步不建任何力，只定索引——但**所有 fail-closed 判据都在这里**，
    后面建力的代码可以假定输入已经合法。
    """
    from openmm import unit

    _assert_supported_forces(system_a, "配体 A 体系")
    _assert_supported_forces(system_b, "配体 B 体系")

    list_a = [int(i) for i in ligand_indices_a]
    list_b = [int(i) for i in ligand_indices_b]
    if len(list_a) != mapping.n_atoms_a:
        raise RBFEHybridBuildError(
            f"ligand_indices_a 有 {len(list_a)} 个原子，映射里 A 有 {mapping.n_atoms_a} 个"
        )
    if len(list_b) != mapping.n_atoms_b:
        raise RBFEHybridBuildError(
            f"ligand_indices_b 有 {len(list_b)} 个原子，映射里 B 有 {mapping.n_atoms_b} 个"
        )
    if len(set(list_a)) != len(list_a) or len(set(list_b)) != len(list_b):
        raise RBFEHybridBuildError("ligand_indices 有重复")

    env_a, _env_b = _assert_environment_matches(system_a, list_a, system_b, list_b)

    a_local_to_hybrid = {local: list_a[local] for local in range(len(list_a))}
    core_b_to_a = {int(b): int(a) for a, b in mapping.core_pairs}

    b_local_to_hybrid = {}
    next_index = system_a.getNumParticles()
    b_only_hybrid = []
    for local_b in range(len(list_b)):
        if local_b in core_b_to_a:
            b_local_to_hybrid[local_b] = a_local_to_hybrid[core_b_to_a[local_b]]
        else:
            b_local_to_hybrid[local_b] = next_index
            b_only_hybrid.append(next_index)
            next_index += 1

    # 🔑 计划 §5.3：「保持跨 λ 相同的粒子、质量和约束结构」。core 原子在两个体系
    # 里必须是同一个粒子，质量不同就不是同一个粒子——元素已由映射验证保证相同，
    # 质量还不同说明输入的力场里有同位素或质量重分配（HMR），首版不处理。
    for local_a, local_b in mapping.core_pairs:
        ma = _quantity_value(system_a.getParticleMass(list_a[local_a]))
        mb = _quantity_value(system_b.getParticleMass(list_b[local_b]))
        if abs(ma - mb) > 1e-6:
            raise RBFEHybridBuildError(
                f"core 原子对（A 分子内 {local_a} / B 分子内 {local_b}）质量不同："
                f"{ma} vs {mb}。同位素或质量重分配（HMR）首版不处理——"
                "跨 λ 粒子质量必须相同（计划 §5.3）。"
            )

    return HybridTopologyLayout(
        n_particles=next_index,
        a_local_to_hybrid=dict(sorted(a_local_to_hybrid.items())),
        b_local_to_hybrid=dict(sorted(b_local_to_hybrid.items())),
        core=tuple(sorted(a_local_to_hybrid[a] for a, _ in mapping.core_pairs)),
        a_only=tuple(sorted(a_local_to_hybrid[i] for i in mapping.a_only)),
        b_only=tuple(sorted(b_only_hybrid)),
        environment=tuple(env_a),
    )




def _b_index_to_hybrid(system_a, ligand_indices_a, system_b, ligand_indices_b, layout) -> dict:
    """`system_b` 的**每个**粒子索引 -> hybrid 索引。

    环境按顺序对应（`_assert_environment_matches` 已保证两侧环境逐位相同），
    配体按冻结映射走。
    """
    set_a = {int(i) for i in ligand_indices_a}
    set_b = {int(i) for i in ligand_indices_b}
    env_a = [i for i in range(system_a.getNumParticles()) if i not in set_a]
    env_b = [i for i in range(system_b.getNumParticles()) if i not in set_b]
    result = {int(ib): int(ia) for ia, ib in zip(env_a, env_b)}
    for local_b, index_b in enumerate(int(i) for i in ligand_indices_b):
        result[index_b] = layout.b_local_to_hybrid[local_b]
    return result


def _bond_rows(system, index_map, alchemical_only=None) -> dict:
    """{(i,j) hybrid 对: (r0, k)}。同一对出现多次即拒绝——不知道该插值哪一个。"""
    import openmm

    rows = {}
    for force in _all_forces_of_type(system, openmm.HarmonicBondForce):
        for i in range(force.getNumBonds()):
            p1, p2, length, k = force.getBondParameters(i)
            key = tuple(sorted((index_map[int(p1)], index_map[int(p2)])))
            value = (_quantity_value(length), _quantity_value(k))
            if key in rows and rows[key] != value:
                raise RBFEHybridBuildError(
                    f"键 {key} 出现多次且参数不同：{rows[key]} vs {value}。"
                    "重复成键项的插值语义未定义，首版拒绝。"
                )
            rows[key] = value
    return rows


def _angle_rows(system, index_map) -> dict:
    import openmm

    rows = {}
    for force in _all_forces_of_type(system, openmm.HarmonicAngleForce):
        for i in range(force.getNumAngles()):
            p1, p2, p3, theta, k = force.getAngleParameters(i)
            a, b, c = (index_map[int(p)] for p in (p1, p2, p3))
            key = (min(a, c), b, max(a, c))
            value = (_quantity_value(theta), _quantity_value(k))
            if key in rows and rows[key] != value:
                raise RBFEHybridBuildError(
                    f"键角 {key} 出现多次且参数不同：{rows[key]} vs {value}，首版拒绝。"
                )
            rows[key] = value
    return rows


def _torsion_rows(system, index_map) -> list:
    """[(四元组, periodicity, phase, k)]。二面角**不去重**：同一四元组的多个
    周期项是正常的力场写法。"""
    import openmm

    rows = []
    for force in _all_forces_of_type(system, openmm.PeriodicTorsionForce):
        for i in range(force.getNumTorsions()):
            p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
            quad = tuple(index_map[int(p)] for p in (p1, p2, p3, p4))
            if quad[0] > quad[-1]:
                quad = tuple(reversed(quad))
            rows.append((quad, int(periodicity), _quantity_value(phase), _quantity_value(k)))
    return rows


#: 力组约定。dummy 的成键项**必须**在自己的力组里，否则端点等价性无法精确验证：
#: λ=0 时 hybrid 比纯 A 体系多出来的，恰好就是 B 侧 dummy 的成键项（可分离因子）。
RBFE_FORCE_GROUP_DEFAULT = 0
RBFE_FORCE_GROUP_DUMMY_A = 1
RBFE_FORCE_GROUP_DUMMY_B = 2


def _add_bonded_forces(hybrid, system_a, system_b, layout, a_map, b_map) -> dict:
    """把成键项装进 hybrid 体系。

    ## 分类规则（首版受限 builder 的核心约定）

    | 力项涉及的原子 | 处理 | 力组 |
    |---|---|---|
    | 纯环境 | 从 A 原样搬 | 0 |
    | 纯 core，A、B 参数相同 | 原样搬 | 0 |
    | 纯 core，A、B 参数不同 | `Custom*Force` 按 `lambda_rbfe_bonded` 插值参数 | 0 |
    | 涉及 A-only（dummy） | **全强度保留，永不缩放** | 1 |
    | 涉及 B-only（dummy） | **全强度保留，永不缩放** | 2 |

    最后两条是 dummy 能抵消的**前提**：dummy 原子的键、角、二面角在两个端点都保持
    全强度，于是它的构型积分是一个与 λ 无关的可分离因子，在 complex 与 solvent
    两腿之间严格相消（计划 §5.3 要求「明确可分离因子／修正及其在热力学循环中的
    抵消条件」）。一旦把 dummy 的成键项也跟着 λ 关掉，这个因子就变成 λ 相关的，
    抵消条件不再成立——OpenFE 官方文档警告的「dummy bonded 处理不抵消导致系统
    误差」正是这个。

    分到独立力组还有一个直接用处：`verify_hybrid_endpoints` 靠扣掉对侧 dummy 力组
    来做**精确**的端点比对，不用估计容差。

    ## 二面角为什么不插值参数而是双项缩放

    键/角的参数是 (k, r0)，A、B 一一对应，插值有明确意义。二面角不同：同一个四元组
    在 A、B 里可能有不同的周期项集合、不同的 phase，硬要配对再插值需要先解决
    「哪个周期项对哪个」，而且 phase 插值本身没有物理意义。所以纯 core 的二面角走
    **双项缩放**：A 的项乘 (1-λ)、B 的项乘 λ。λ=0 只剩 A 的、λ=1 只剩 B 的，
    端点严格正确，中间平滑，且完全不需要配对。
    """
    import openmm

    core = set(layout.core)
    a_only, b_only = set(layout.a_only), set(layout.b_only)
    alchemical = core | a_only | b_only

    plain_bond = openmm.HarmonicBondForce()
    plain_angle = openmm.HarmonicAngleForce()
    plain_torsion = openmm.PeriodicTorsionForce()
    dummy_forces = {}
    for side, group in (("A", RBFE_FORCE_GROUP_DUMMY_A), ("B", RBFE_FORCE_GROUP_DUMMY_B)):
        bond = openmm.HarmonicBondForce()
        angle = openmm.HarmonicAngleForce()
        torsion = openmm.PeriodicTorsionForce()
        for force in (bond, angle, torsion):
            force.setForceGroup(group)
        dummy_forces[side] = {"bond": bond, "angle": angle, "torsion": torsion}

    interp_bond = openmm.CustomBondForce(
        f"0.5*k*(r-r0)^2;"
        f" k=(1-{LAMBDA_BONDED})*kA + {LAMBDA_BONDED}*kB;"
        f" r0=(1-{LAMBDA_BONDED})*r0A + {LAMBDA_BONDED}*r0B"
    )
    for name in ("r0A", "kA", "r0B", "kB"):
        interp_bond.addPerBondParameter(name)
    interp_bond.addGlobalParameter(LAMBDA_BONDED, 0.0)

    interp_angle = openmm.CustomAngleForce(
        f"0.5*k*(theta-theta0)^2;"
        f" k=(1-{LAMBDA_BONDED})*kA + {LAMBDA_BONDED}*kB;"
        f" theta0=(1-{LAMBDA_BONDED})*t0A + {LAMBDA_BONDED}*t0B"
    )
    for name in ("t0A", "kA", "t0B", "kB"):
        interp_angle.addPerAngleParameter(name)
    interp_angle.addGlobalParameter(LAMBDA_BONDED, 0.0)

    torsion_a = openmm.CustomTorsionForce(
        f"(1-{LAMBDA_BONDED})*k*(1+cos(periodicity*theta-phase))"
    )
    torsion_b = openmm.CustomTorsionForce(
        f"{LAMBDA_BONDED}*k*(1+cos(periodicity*theta-phase))"
    )
    for force in (torsion_a, torsion_b):
        for name in ("periodicity", "phase", "k"):
            force.addPerTorsionParameter(name)
        force.addGlobalParameter(LAMBDA_BONDED, 0.0)

    bonds_a, bonds_b = _bond_rows(system_a, a_map), _bond_rows(system_b, b_map)
    angles_a, angles_b = _angle_rows(system_a, a_map), _angle_rows(system_b, b_map)

    # 🔑 前置条件：core–core 的**键连接必须完全一致**。不一致意味着公共核心内部有
    # 键生成/断裂——`validate_mapping` 的连通性判据管不到这个（它只查 core 是一整块，
    # 不查 A、B 的 core 内部连法是否相同）。首版拒绝。
    core_bonds_a = {key for key in bonds_a if set(key) <= core}
    core_bonds_b = {key for key in bonds_b if set(key) <= core}
    if core_bonds_a != core_bonds_b:
        raise RBFEHybridBuildError(
            "公共核心内部的键连接在 A、B 里不同："
            f"只在 A 里的 {sorted(core_bonds_a - core_bonds_b)}、"
            f"只在 B 里的 {sorted(core_bonds_b - core_bonds_a)}。"
            "这是 core 内部的成键/断键，首版拒绝（计划 §2）。"
        )

    stats = {"plain_bond": 0, "interp_bond": 0, "dummy_bond_A": 0, "dummy_bond_B": 0,
             "plain_angle": 0, "interp_angle": 0, "dummy_angle_A": 0, "dummy_angle_B": 0,
             "plain_torsion": 0, "torsion_A_scaled": 0, "torsion_B_scaled": 0,
             "dummy_torsion_A": 0, "dummy_torsion_B": 0, "env_torsion": 0}

    def dummy_side(atoms) -> Optional[str]:
        if atoms & a_only:
            return "A"
        if atoms & b_only:
            return "B"
        return None

    # --- 键 ---
    for key in sorted(set(bonds_a) | set(bonds_b)):
        atoms = set(key)
        side = dummy_side(atoms)
        if not (atoms & alchemical):
            r0, k = bonds_a[key]
            plain_bond.addBond(key[0], key[1], r0, k)
            stats["plain_bond"] += 1
        elif side is not None:
            source = bonds_a.get(key, bonds_b.get(key))
            dummy_forces[side]["bond"].addBond(key[0], key[1], source[0], source[1])
            stats[f"dummy_bond_{side}"] += 1
        else:
            row_a, row_b = bonds_a[key], bonds_b[key]
            if row_a == row_b:
                plain_bond.addBond(key[0], key[1], row_a[0], row_a[1])
                stats["plain_bond"] += 1
            else:
                interp_bond.addBond(key[0], key[1], [row_a[0], row_a[1], row_b[0], row_b[1]])
                stats["interp_bond"] += 1

    # --- 键角 ---
    for key in sorted(set(angles_a) | set(angles_b)):
        atoms = {key[0], key[1], key[2]}
        side = dummy_side(atoms)
        if not (atoms & alchemical):
            t0, k = angles_a[key]
            plain_angle.addAngle(key[0], key[1], key[2], t0, k)
            stats["plain_angle"] += 1
        elif side is not None:
            source = angles_a.get(key, angles_b.get(key))
            dummy_forces[side]["angle"].addAngle(key[0], key[1], key[2], source[0], source[1])
            stats[f"dummy_angle_{side}"] += 1
        else:
            row_a, row_b = angles_a.get(key), angles_b.get(key)
            if row_a is None or row_b is None:
                missing = "B" if row_a is not None else "A"
                raise RBFEHybridBuildError(
                    f"纯 core 的键角 {key} 在 {missing} 侧不存在。core 内部连接已校验"
                    "相同，出现这种情况说明力场对同一连接给了不同的键角项集合——"
                    "首版不猜该插到什么，拒绝。"
                )
            if row_a == row_b:
                plain_angle.addAngle(key[0], key[1], key[2], row_a[0], row_a[1])
                stats["plain_angle"] += 1
            else:
                interp_angle.addAngle(
                    key[0], key[1], key[2], [row_a[0], row_a[1], row_b[0], row_b[1]]
                )
                stats["interp_angle"] += 1

    # --- 二面角 ---
    #
    # 🔑 纯 core 且 A、B **逐位相同**的二面角走 native `PeriodicTorsionForce`，
    # 不塞进 `CustomTorsionForce`。原因不是省力：OpenMM 内建的二面角力对**退化几何**
    # （四个原子共线，二面角没有定义）是安全的，而 `CustomTorsionForce` 的解析导数
    # 在那里给 NaN。生产轨迹里出现瞬时近共线不是不可能，所以能用 native 的就用
    # native，把 Custom 的暴露面压到"真正随 λ 变的那几个"。
    rows_a = _torsion_rows(system_a, a_map)
    rows_b = _torsion_rows(system_b, b_map)
    core_torsions_a = {}
    core_torsions_b = {}

    def bucket(rows, sink, other_side_marker):
        for quad, periodicity, phase, k in rows:
            atoms = set(quad)
            side = dummy_side(atoms)
            if not (atoms & alchemical):
                if other_side_marker == "A":            # 环境项只从 A 搬一次
                    plain_torsion.addTorsion(*quad, periodicity, phase, k)
                    stats["env_torsion"] += 1
                continue
            if side is not None:
                dummy_forces[side]["torsion"].addTorsion(*quad, periodicity, phase, k)
                stats[f"dummy_torsion_{side}"] += 1
                continue
            key = (quad, int(periodicity), round(float(phase), 12))
            sink.setdefault(key, []).append(float(k))

    bucket(rows_a, core_torsions_a, "A")
    bucket(rows_b, core_torsions_b, "B")

    for key in sorted(set(core_torsions_a) | set(core_torsions_b), key=repr):
        quad, periodicity, phase = key
        ks_a = sorted(core_torsions_a.get(key, []))
        ks_b = sorted(core_torsions_b.get(key, []))
        shared = 0
        if ks_a == ks_b:                                # A、B 完全一样 → native
            for k in ks_a:
                plain_torsion.addTorsion(*quad, periodicity, phase, k)
                stats["plain_torsion"] += 1
            continue
        for k in ks_a:
            torsion_a.addTorsion(*quad, [float(periodicity), phase, k])
            stats["torsion_A_scaled"] += 1
        for k in ks_b:
            torsion_b.addTorsion(*quad, [float(periodicity), phase, k])
            stats["torsion_B_scaled"] += 1

    for force in (plain_bond, plain_angle, plain_torsion,
                  interp_bond, interp_angle, torsion_a, torsion_b):
        hybrid.addForce(force)
    for side in ("A", "B"):
        for force in dummy_forces[side].values():
            hybrid.addForce(force)
    return stats


def _add_constraints(hybrid, system_a, system_b, layout, a_map, b_map) -> int:
    """约束：环境与 A 侧照搬，B-only 的从 B 搬；core–core 约束两边必须一致。

    计划 §5.3：「保持跨 λ 相同的粒子、质量和**约束结构**」。约束距离没法随 λ
    插值（约束是硬的），所以 core–core 约束只要 A、B 不同就拒绝。
    """
    core = set(layout.core)
    b_only = set(layout.b_only)

    rows_a = {}
    for i in range(system_a.getNumConstraints()):
        p1, p2, distance = system_a.getConstraintParameters(i)
        rows_a[tuple(sorted((a_map[int(p1)], a_map[int(p2)])))] = _quantity_value(distance)
    rows_b = {}
    for i in range(system_b.getNumConstraints()):
        p1, p2, distance = system_b.getConstraintParameters(i)
        rows_b[tuple(sorted((b_map[int(p1)], b_map[int(p2)])))] = _quantity_value(distance)

    for key in set(rows_a) & set(rows_b):
        if set(key) <= core and abs(rows_a[key] - rows_b[key]) > 1e-9:
            raise RBFEHybridBuildError(
                f"core–core 约束 {key} 的距离在 A、B 里不同：{rows_a[key]} vs {rows_b[key]}。"
                "约束是硬的，没法随 λ 插值，首版拒绝（计划 §5.3）。"
            )
    merged = dict(rows_a)
    for key, distance in rows_b.items():
        if key in merged:
            continue
        if not (set(key) & b_only):
            raise RBFEHybridBuildError(
                f"约束 {key} 只存在于 B 侧，但它不涉及任何 B-only 原子。"
                "跨 λ 约束结构必须一致（计划 §5.3），首版拒绝。"
            )
        merged[key] = distance

    for (p1, p2), distance in sorted(merged.items()):
        hybrid.addConstraint(p1, p2, distance)
    return len(merged)




#: Beutler softcore 的无量纲 α，与 `abfe_core.BeutlerSoftcoreBuilder` 同一约定
#: （表达式内乘 sigma^6，不是绝对 nm^6）。
RBFE_SOFTCORE_ALPHA = 0.5


def _particle_endpoint_params(system_a, system_b, layout, a_map, b_map) -> dict:
    """每个 hybrid 粒子在两个端点的 (q, sigma, epsilon)。

    dummy 侧一律取 (0, 对侧的 sigma, 0)：**sigma 不取 0**——softcore 的分母里
    有 sigma^6，取 0 会让软化项整个消失，dummy 在消失过程中反而变成硬球。
    """
    nb_a, nb_b = _nb_force_of(system_a, "配体 A 体系"), _nb_force_of(system_b, "配体 B 体系")
    hybrid_a = {a_map[i]: i for i in range(system_a.getNumParticles())}
    hybrid_b = {b_map[i]: i for i in range(system_b.getNumParticles())}

    params = {}
    for h in range(layout.n_particles):
        side_a = hybrid_a.get(h)
        side_b = hybrid_b.get(h)
        if side_a is not None:
            qa, sa, ea = (_quantity_value(v) for v in nb_a.getParticleParameters(side_a))
        else:
            qa, sa, ea = None, None, None
        if side_b is not None:
            qb, sb, eb = (_quantity_value(v) for v in nb_b.getParticleParameters(side_b))
        else:
            qb, sb, eb = None, None, None
        if side_a is None:                       # B-only
            qa, sa, ea = 0.0, sb, 0.0
        if side_b is None:                       # A-only
            qb, sb, eb = 0.0, sa, 0.0
        params[h] = {"qA": qa, "sigA": sa, "epsA": ea, "qB": qb, "sigB": sb, "epsB": eb}
    return params


def _exception_rows(system, index_map) -> dict:
    import openmm

    nb = _nb_force_of(system, "体系")
    rows = {}
    for i in range(nb.getNumExceptions()):
        p1, p2, charge_prod, sigma, epsilon = nb.getExceptionParameters(i)
        key = tuple(sorted((index_map[int(p1)], index_map[int(p2)])))
        rows[key] = (
            _quantity_value(charge_prod),
            _quantity_value(sigma),
            _quantity_value(epsilon),
        )
    return rows


def _add_nonbonded_forces(hybrid, system_a, system_b, layout, a_map, b_map) -> dict:
    """非键层。

    ## 静电：全部留在 native `NonbondedForce`，用参数 offset 走 λ

    这是从 ABFE 抄的最关键一条（`ibs_engine.configure_pme_ligand_charge_offsets`
    的做法）：**不要**把配体静电搬进 `CustomNonbondedForce`——那会把库仑在 cutoff
    处截断，PME 倒空间就不对了。`addParticleParameterOffset` 让 OpenMM 自己在
    正空间和倒空间里都按 q(λ)=base+λ·scale 处理。

    系数按 RBFE 端点写：core 是 `base=qᴬ, scale=qᴮ−qᴬ`；A-only 是 `base=qᴬ,
    scale=−qᴬ`；B-only 是 `base=0, scale=qᴮ`。（ABFE 那边写的是 `base=0,
    scale=q`，插到零——那是 ABFE 的端点，照抄就错。）

    ## LJ：从 native 力上摘掉，拆成三个 CustomNonbondedForce

    | 力 | 覆盖的对 | λ 语义 |
    |---|---|---|
    | core | core×core、core×env | 参数插值，**不加 softcore**（两端都是实原子） |
    | A-only | A-only×(自身/core/env) | 前因子 (1−λ)，softcore lift λ |
    | B-only | B-only×(自身/core/env) | 前因子 λ，softcore lift (1−λ) |

    **A-only × B-only 不在任何 interaction group 里，也显式加了排除**——两组
    dummy 永远不能相互看见，这是 hybrid 拓扑的硬约束，ABFE 里没有对应概念。

    env×env 的 LJ 仍然留在 native 力上（连同它的长程校正），所以炼金原子在 native
    力上的 epsilon 被清零。代价见返回值里的 `alchemical_lj_lrc_included=False`。
    """
    import openmm
    from openmm import unit

    nb_a = _nb_force_of(system_a, "配体 A 体系")
    endpoint = _particle_endpoint_params(system_a, system_b, layout, a_map, b_map)
    core, a_only, b_only = set(layout.core), set(layout.a_only), set(layout.b_only)
    env = set(layout.environment)
    alchemical = core | a_only | b_only

    # ---------------- native NonbondedForce（静电 + env LJ） ----------------
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(nb_a.getNonbondedMethod())
    nb.setCutoffDistance(nb_a.getCutoffDistance())
    nb.setUseSwitchingFunction(nb_a.getUseSwitchingFunction())
    nb.setSwitchingDistance(nb_a.getSwitchingDistance())
    nb.setUseDispersionCorrection(nb_a.getUseDispersionCorrection())
    nb.setEwaldErrorTolerance(nb_a.getEwaldErrorTolerance())
    nb.setReactionFieldDielectric(nb_a.getReactionFieldDielectric())
    alpha, nx, ny, nz = nb_a.getPMEParameters()
    nb.setPMEParameters(alpha, nx, ny, nz)
    nb.addGlobalParameter(LAMBDA_CHARGE, 0.0)

    for h in range(layout.n_particles):
        row = endpoint[h]
        epsilon = 0.0 if h in alchemical else row["epsA"]
        nb.addParticle(
            row["qA"] * unit.elementary_charge,
            row["sigA"] * unit.nanometer,
            epsilon * unit.kilojoule_per_mole,
        )
        if h in alchemical:
            delta_q = row["qB"] - row["qA"]
            nb.addParticleParameterOffset(
                LAMBDA_CHARGE,
                h,
                delta_q * unit.elementary_charge,
                0.0 * unit.nanometer,
                0.0 * unit.kilojoule_per_mole,
            )

    # ---------------- exception / exclusion ----------------
    exc_a = _exception_rows(system_a, a_map)
    exc_b = _exception_rows(system_b, b_map)

    core_exc_a = {k for k in exc_a if set(k) <= core}
    core_exc_b = {k for k in exc_b if set(k) <= core}
    if core_exc_a != core_exc_b:
        raise RBFEHybridBuildError(
            "公共核心内部的 exception（1-2/1-3 排除与 1-4）在 A、B 里不一致："
            f"只在 A 里的 {sorted(core_exc_a - core_exc_b)}、"
            f"只在 B 里的 {sorted(core_exc_b - core_exc_a)}。"
            "同一份 core 连接却给出不同的排除表，首版拒绝。"
        )

    exception_pairs = sorted(set(exc_a) | set(exc_b))
    # A-only × B-only：两组 dummy 永不相互作用
    forbidden_pairs = sorted(
        (min(x, y), max(x, y)) for x in a_only for y in b_only
    )
    all_excluded = sorted(set(exception_pairs) | set(forbidden_pairs))

    pair14 = []
    for key in exception_pairs:
        row_a, row_b = exc_a.get(key), exc_b.get(key)
        cp_a, sig_a, eps_a = row_a if row_a else (0.0, (row_b or (0, 0.3, 0))[1], 0.0)
        cp_b, sig_b, eps_b = row_b if row_b else (0.0, sig_a, 0.0)
        is_alchemical = bool(set(key) & alchemical)
        native_eps = 0.0 if is_alchemical else eps_a
        index = nb.addException(
            key[0],
            key[1],
            cp_a * unit.elementary_charge**2,
            sig_a * unit.nanometer,
            native_eps * unit.kilojoule_per_mole,
        )
        if is_alchemical:
            nb.addExceptionParameterOffset(
                LAMBDA_CHARGE,
                index,
                (cp_b - cp_a) * unit.elementary_charge**2,
                0.0 * unit.nanometer,
                0.0 * unit.kilojoule_per_mole,
            )
            if eps_a != 0.0 or eps_b != 0.0:
                pair14.append((key, sig_a or 0.1, eps_a, sig_b or 0.1, eps_b))
    for key in forbidden_pairs:
        nb.addException(
            key[0], key[1],
            0.0 * unit.elementary_charge**2,
            0.3 * unit.nanometer,
            0.0 * unit.kilojoule_per_mole,
        )
    hybrid.addForce(nb)

    # ---------------- LJ：三个 CustomNonbondedForce ----------------
    def new_lj_force(expression) -> "openmm.CustomNonbondedForce":
        force = openmm.CustomNonbondedForce(expression)
        for name in ("sigA", "epsA", "sigB", "epsB"):
            force.addPerParticleParameter(name)
        force.addGlobalParameter(LAMBDA_STERICS, 0.0)
        for h in range(layout.n_particles):
            row = endpoint[h]
            force.addParticle([row["sigA"], row["epsA"], row["sigB"], row["epsB"]])
        method = (
            openmm.CustomNonbondedForce.CutoffPeriodic
            if nb_a.getNonbondedMethod() in (
                openmm.NonbondedForce.PME,
                openmm.NonbondedForce.Ewald,
                openmm.NonbondedForce.CutoffPeriodic,
            )
            else openmm.CustomNonbondedForce.CutoffNonPeriodic
        )
        force.setNonbondedMethod(method)
        force.setCutoffDistance(nb_a.getCutoffDistance())
        force.setUseSwitchingFunction(nb_a.getUseSwitchingFunction())
        if nb_a.getUseSwitchingFunction():
            force.setSwitchingDistance(nb_a.getSwitchingDistance())
        # 长程校正一律关掉：softcore/插值参数下 OpenMM 的解析尾项公式不成立。
        # ABFE 那边是另算 `_lj_softcore_tail_radial_integrals`；R1b v1 不做，
        # 如实记在 provenance 里。
        force.setUseLongRangeCorrection(False)
        for p1, p2 in all_excluded:
            force.addExclusion(p1, p2)
        return force

    ls = LAMBDA_STERICS
    force_core = new_lj_force(
        "4*epsij*((sigij/r)^12-(sigij/r)^6);"
        " sigij=0.5*(si+sj);"
        " epsij=sqrt(ei*ej);"
        f" si=(1-{ls})*sigA1+{ls}*sigB1;"
        f" sj=(1-{ls})*sigA2+{ls}*sigB2;"
        f" ei=(1-{ls})*epsA1+{ls}*epsB1;"
        f" ej=(1-{ls})*epsA2+{ls}*epsB2"
    )
    force_a = new_lj_force(
        f"(1-{ls})*4*epsij*(sigij^12/D^2 - sigij^6/D);"
        f" D=r^6 + {RBFE_SOFTCORE_ALPHA}*sigij^6*{ls};"
        " sigij=0.5*(sigA1+sigA2);"
        " epsij=sqrt(epsA1*epsA2)"
    )
    force_b = new_lj_force(
        f"{ls}*4*epsij*(sigij^12/D^2 - sigij^6/D);"
        f" D=r^6 + {RBFE_SOFTCORE_ALPHA}*sigij^6*(1-{ls});"
        " sigij=0.5*(sigB1+sigB2);"
        " epsij=sqrt(epsB1*epsB2)"
    )

    core_and_env = core | env
    group_counts = {"core": 0, "a_only": 0, "b_only": 0}
    if core:
        force_core.addInteractionGroup(core, core)
        group_counts["core"] += 1
        if env:
            force_core.addInteractionGroup(core, env)
            group_counts["core"] += 1
    if a_only:
        force_a.addInteractionGroup(a_only, a_only)
        group_counts["a_only"] += 1
        if core_and_env:
            force_a.addInteractionGroup(a_only, core_and_env)
            group_counts["a_only"] += 1
    if b_only:
        force_b.addInteractionGroup(b_only, b_only)
        group_counts["b_only"] += 1
        if core_and_env:
            force_b.addInteractionGroup(b_only, core_and_env)
            group_counts["b_only"] += 1

    # 🔑 **一个 interaction group 都没有的 CustomNonbondedForce 会计算全体粒子对。**
    # 这是 OpenMM 的默认行为，不是"什么都不算"。所以 A-only 或 B-only 为空时
    # （比如 A→A 自边，或者任何"只改参数、不增删原子"的变换——那是很常见的一类边），
    # 空着的那个力会退化成对全部粒子求和，与 force_core 和 native 力**重复计数**。
    #
    # 这个 bug 一度被"正确答案"掩盖：A→A 的 ΔG 仍然算出 0，因为 λ 表对称、两个端点
    # 上 softcore lift 都为 0，两个力恰好互为镜像而抵消——**端点对了，中间态全错**。
    # 是 `test_self_edge_u_kn_rows_are_identical`（验中间态而不是验端点）抓到的。
    for label, force in (("core", force_core), ("a_only", force_a), ("b_only", force_b)):
        if group_counts[label] > 0:
            hybrid.addForce(force)
    nonbonded_group_counts = dict(group_counts)

    # ---------------- 1-4 的 LJ（从 native 摘出来的那部分） ----------------
    if pair14:
        f14 = openmm.CustomBondForce(
            f"(1-{ls})*4*epsA*((sigA/r)^12-(sigA/r)^6)"
            f" + {ls}*4*epsB*((sigB/r)^12-(sigB/r)^6)"
        )
        for name in ("sigA", "epsA", "sigB", "epsB"):
            f14.addPerBondParameter(name)
        f14.addGlobalParameter(LAMBDA_STERICS, 0.0)
        for (p1, p2), sig_a, eps_a, sig_b, eps_b in pair14:
            f14.addBond(p1, p2, [sig_a, eps_a, sig_b, eps_b])
        hybrid.addForce(f14)

    return {
        "n_exceptions": len(exception_pairs),
        "n_forbidden_dummy_pairs": len(forbidden_pairs),
        "lj_interaction_groups": nonbonded_group_counts,
        "lj_forces_added": [
            label for label, count in nonbonded_group_counts.items() if count > 0
        ],
        "n_lj14_terms": len(pair14),
        "native_dispersion_correction": bool(nb_a.getUseDispersionCorrection()),
        "alchemical_lj_lrc_included": False,
        "softcore_alpha": RBFE_SOFTCORE_ALPHA,
    }




@dataclass(frozen=True)
class HybridLambdaSchedule:
    """λ 路径。计划 §2 要求 **λ 协议显式配置，不沿用隐藏默认值**——所以这个对象
    必须由调用方显式构造，builder 不提供"没给就用默认"的行为。

    三条 λ 分开是因为它们的物理含义不同（计划 §5.3：「路径至少描述 charge、
    sterics 和 bonded 项」），而且分段切换会改变中间态的有效总电荷——那正是
    `hybrid_charge_ledger` 要盯的东西。
    """

    name: str
    charge: tuple
    sterics: tuple
    bonded: tuple

    def __post_init__(self) -> None:
        lengths = {len(self.charge), len(self.sterics), len(self.bonded)}
        if len(lengths) != 1:
            raise RBFEHybridBuildError(
                f"三条 λ 的态数不一致：charge={len(self.charge)}、"
                f"sterics={len(self.sterics)}、bonded={len(self.bonded)}"
            )
        if self.n_states < 2:
            raise RBFEHybridBuildError(f"λ 表至少需要 2 个态（端点 A 与 B）：收到 {self.n_states}")
        for label, values in (("charge", self.charge), ("sterics", self.sterics), ("bonded", self.bonded)):
            if any(not (0.0 <= float(v) <= 1.0) for v in values):
                raise RBFEHybridBuildError(f"λ_{label} 有值不在 [0, 1] 内：{values}")
            if float(values[0]) != 0.0 or float(values[-1]) != 1.0:
                raise RBFEHybridBuildError(
                    f"λ_{label} 的端点必须是 0 和 1（λ=0 对应 A、λ=1 对应 B，计划 §3）："
                    f"收到 {values[0]} … {values[-1]}"
                )
            if any(float(b) < float(a) for a, b in zip(values, values[1:])):
                raise RBFEHybridBuildError(f"λ_{label} 必须单调不减：{values}")

    @property
    def n_states(self) -> int:
        return len(self.charge)

    def state(self, index: int) -> dict:
        return {
            LAMBDA_CHARGE: float(self.charge[index]),
            LAMBDA_STERICS: float(self.sterics[index]),
            LAMBDA_BONDED: float(self.bonded[index]),
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_states": self.n_states,
            LAMBDA_CHARGE: [float(v) for v in self.charge],
            LAMBDA_STERICS: [float(v) for v in self.sterics],
            LAMBDA_BONDED: [float(v) for v in self.bonded],
        }

    @classmethod
    def linear(cls, n_states: int) -> "HybridLambdaSchedule":
        """三条 λ 同步线性推进。最简单、也最容易在端点附近出现采样困难的一条。"""
        if n_states < 2:
            raise RBFEHybridBuildError(f"n_states 至少为 2：收到 {n_states}")
        values = tuple(i / (n_states - 1) for i in range(n_states))
        return cls(name=f"rbfe_linear_{n_states}", charge=values, sterics=values, bonded=values)

    @classmethod
    def charge_then_sterics(cls, n_charge: int, n_sterics: int) -> "HybridLambdaSchedule":
        """先走完电荷，再走 sterics 与 bonded。

        ⚠ 这条路径的中间态**有效总电荷不恒定**（电荷已经变成 B 的，而 B-only 原子
        的 sterics 还没打开）。这不是错误——路径可以自选——但它正是计划 §5.3 点名
        要核对的情形，`hybrid_charge_ledger` 会把逐 λ 的总电荷列出来。
        """
        if n_charge < 2 or n_sterics < 2:
            raise RBFEHybridBuildError("n_charge 与 n_sterics 都至少为 2")
        charge = tuple(i / (n_charge - 1) for i in range(n_charge)) + (1.0,) * (n_sterics - 1)
        rest = (0.0,) * (n_charge - 1) + tuple(i / (n_sterics - 1) for i in range(n_sterics))
        return cls(
            name=f"rbfe_charge{n_charge}_then_sterics{n_sterics}",
            charge=charge, sterics=rest, bonded=rest,
        )


@dataclass(frozen=True)
class HybridSystemBundle:
    """`build_hybrid_system` 的产物：System + 布局 + λ 表 + 可审计的溯源。"""

    system: object
    layout: HybridTopologyLayout
    schedule: HybridLambdaSchedule
    mapping_fingerprint: str
    protocol_version: int
    provenance: dict

    def apply_lambda_state(self, context, index: int) -> dict:
        """把第 index 个 λ 态写进 Context。返回实际写入的值，便于落盘对账。"""
        values = self.schedule.state(index)
        for name, value in values.items():
            context.setParameter(name, value)
        return values

    def fingerprint(self) -> str:
        payload = {
            "hybrid_protocol_version": int(self.protocol_version),
            "mapping_fingerprint": self.mapping_fingerprint,
            "schedule": self.schedule.to_dict(),
            "layout": self.layout.to_dict(),
            "bonded": self.provenance.get("bonded"),
            "nonbonded": self.provenance.get("nonbonded"),
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "hybrid_protocol_version": int(self.protocol_version),
            "fingerprint": self.fingerprint(),
            "mapping_fingerprint": self.mapping_fingerprint,
            "layout": self.layout.to_dict(),
            "schedule": self.schedule.to_dict(),
            "provenance": self.provenance,
        }


def build_hybrid_system(
    system_a,
    ligand_indices_a,
    system_b,
    ligand_indices_b,
    mapping: AtomMapping,
    schedule: HybridLambdaSchedule,
) -> HybridSystemBundle:
    """R1b：由 A、B 两个体系 + 冻结映射构建 hybrid System。

    输入约定：`system_a` 与 `system_b` 是**同一个环境**里分别装了配体 A、B 的
    完整体系（`_assert_environment_matches` 会逐位核对）。`ligand_indices_*` 按
    **分子内顺序**给出，即第 i 个元素就是映射里分子内索引 i 的那个原子。

    产物的原子编号：环境与配体 A 保持 `system_a` 的原索引，B-only 追加在末尾。
    """
    import openmm

    layout = build_hybrid_layout(system_a, ligand_indices_a, system_b, ligand_indices_b, mapping)
    a_map = {i: i for i in range(system_a.getNumParticles())}
    b_map = _b_index_to_hybrid(system_a, ligand_indices_a, system_b, ligand_indices_b, layout)

    hybrid = openmm.System()
    for i in range(system_a.getNumParticles()):
        hybrid.addParticle(system_a.getParticleMass(i))
    hybrid_to_b = {v: k for k, v in b_map.items()}
    for h in layout.b_only:
        hybrid.addParticle(system_b.getParticleMass(hybrid_to_b[h]))
    hybrid.setDefaultPeriodicBoxVectors(*system_a.getDefaultPeriodicBoxVectors())

    n_constraints = _add_constraints(hybrid, system_a, system_b, layout, a_map, b_map)
    bonded_stats = _add_bonded_forces(hybrid, system_a, system_b, layout, a_map, b_map)
    nonbonded_stats = _add_nonbonded_forces(hybrid, system_a, system_b, layout, a_map, b_map)

    for force in system_a.getForces():
        if isinstance(force, openmm.CMMotionRemover):
            hybrid.addForce(openmm.CMMotionRemover(force.getFrequency()))

    provenance = {
        "builder": "rbfe_core.build_hybrid_system",
        "restricted_builder_version": int(RBFE_HYBRID_PROTOCOL_VERSION),
        "lambda_parameter_names": list(RBFE_LAMBDA_NAMES),
        "n_constraints": int(n_constraints),
        "bonded": bonded_stats,
        "nonbonded": nonbonded_stats,
        "force_groups": {
            "default": RBFE_FORCE_GROUP_DEFAULT,
            "dummy_bonded_A": RBFE_FORCE_GROUP_DUMMY_A,
            "dummy_bonded_B": RBFE_FORCE_GROUP_DUMMY_B,
        },
        "known_gaps": [
            "炼金区的 LJ 长程校正（LRC）未计入：三个 custom LJ 力都关了 "
            "setUseLongRangeCorrection，native 力上炼金原子的 epsilon 已清零。"
            "ABFE 侧有现成的 ibs_engine._lj_softcore_tail_radial_integrals 可接，R1b v1 未接。",
            "手性/结合姿势/互变异构仍未检查（需要坐标，见 validate_mapping 的 unchecked）。",
        ],
    }
    return HybridSystemBundle(
        system=hybrid,
        layout=layout,
        schedule=schedule,
        mapping_fingerprint=mapping.fingerprint(),
        protocol_version=RBFE_HYBRID_PROTOCOL_VERSION,
        provenance=provenance,
    )


# -- R1b 的验收：端点等价 / 有限差分力 / 逐 λ 电荷 -------------------------


def _energy_and_forces(system, positions, parameters=None, *, platform_name="Reference", groups=None):
    import openmm
    from openmm import unit

    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    for name, value in (parameters or {}).items():
        context.setParameter(name, value)
    kwargs = {"getEnergy": True, "getForces": True}
    if groups is not None:
        kwargs["groups"] = groups
    state = context.getState(**kwargs)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    del context, integrator
    energy = float(energy)
    forces = np.asarray(forces, dtype=float)

    # 🔑 NaN 必须在这里炸掉。踩过一次：`max(0.0, nan)` 在 Python 里返回 **0.0**
    # （因为 `nan > 0.0` 是 False），于是端点比对里一个 NaN 力偏差被静默当成"零偏差"
    # 通过了验收。fail-closed 的检查器自己漏 NaN，比没有检查器更危险。
    if not math.isfinite(energy) or not np.all(np.isfinite(forces)):
        bad = sorted(set(np.nonzero(~np.isfinite(forces))[0].tolist()))
        raise RBFEHybridBuildError(
            f"能量或力出现非有限值：energy={energy}，非有限力的原子 {bad[:12]}"
            f"{'…' if len(bad) > 12 else ''}。"
            "常见原因是几何退化（例如二面角的四个原子共线，CustomTorsionForce "
            "的解析导数在那里没有定义），或 softcore 分母被写成 0。"
        )
    return energy, forces


def _without_dispersion_correction(system):
    """深拷一份关掉色散长程校正的体系，只用于**受控比对**。

    LJ 的长程校正是一个只依赖盒体积与粒子参数的**常数**（力恒为 0）。炼金原子的
    epsilon 在 hybrid 的 native 力上被清零之后，native 的 LRC 就不再包含配体那部分
    ——于是 hybrid 与纯 A 体系之间会差一个常数。这个差值是**已知缺口**，不是接线
    错误：力逐原子精确相等就是证据。

    所以端点等价性分两步报：关掉 LRC 做**精确**比对（这才测得到 builder 自己写的
    东西），再单独把 LRC 缺口**量化成一个数**报出来，不留一句含糊的 caveat。
    """
    import openmm

    copy = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    copy.thisown = 1
    for force in copy.getForces():
        if isinstance(force, openmm.NonbondedForce):
            force.setUseDispersionCorrection(False)
        elif isinstance(force, openmm.CustomNonbondedForce):
            force.setUseLongRangeCorrection(False)
    return copy


def verify_hybrid_endpoints(
    bundle: HybridSystemBundle,
    system_a,
    ligand_indices_a,
    system_b,
    ligand_indices_b,
    positions_hybrid,
    *,
    platform_name: str = "Reference",
    energy_tolerance_kj_per_mol: float = 1e-4,
    force_tolerance_kj_per_mol_nm: float = 1e-3,
) -> dict:
    """计划 §5.3：「验证 λ 端点的物理相互作用恢复」。

    ## 判据是精确的，不是"看着像"

    λ=0 时 hybrid 与纯 A 体系相比，**多出来的恰好是 B 侧 dummy 的成键项**——
    B-only 原子此时电荷为 0、sterics 前因子为 0、且与 A-only 之间已被排除，
    所以它对体系的唯一贡献就是自己那几个全强度成键项。那些项被单独放在力组
    `RBFE_FORCE_GROUP_DUMMY_B` 里，于是判据可以写成等式：

        E_hybrid(λ=0) − E[力组 dummy_B] == E_A
        E_hybrid(λ=1) − E[力组 dummy_A] == E_B

    力也一样，逐原子比。这就是计划 §5.3 说的「明确可分离因子」——这里它不仅被
    明确了，还被独立成一个可以直接读数的力组。

    ## 为什么比对时把 LJ 长程校正关掉

    炼金原子的 epsilon 在 hybrid 的 native 力上被清零，native 的色散长程校正因此
    不再包含配体那部分——hybrid 与纯 A 体系之间会差一个**常数**（LRC 只依赖盒体积
    与粒子参数，力恒为 0）。那是**已知缺口**，不是接线错误，所以：等式判据在关掉
    LRC 的副本上做（这才测得到 builder 自己写的东西），缺口本身单独量化成
    `lj_lrc_gap_kJ_per_mol` 报出来。
    """
    from openmm import unit

    layout = bundle.layout
    a_map = {i: i for i in range(system_a.getNumParticles())}
    b_map = _b_index_to_hybrid(system_a, ligand_indices_a, system_b, ligand_indices_b, layout)

    positions = np.asarray(
        [[v.x, v.y, v.z] if hasattr(v, "x") else list(v) for v in positions_hybrid], dtype=float
    )
    if positions.shape[0] != layout.n_particles:
        raise RBFEHybridBuildError(
            f"positions_hybrid 有 {positions.shape[0]} 个原子，hybrid 体系有 "
            f"{layout.n_particles} 个"
        )
    positions_q = positions * unit.nanometer

    positions_a = positions[: system_a.getNumParticles()] * unit.nanometer
    order_b = [b_map[i] for i in range(system_b.getNumParticles())]
    positions_b = positions[order_b] * unit.nanometer

    hybrid_no_lrc = _without_dispersion_correction(bundle.system)

    report = {"endpoints": {}, "passed": True}
    for label, lam, other_group, reference_system, reference_positions, index_map in (
        ("lambda=0 (A)", 0.0, RBFE_FORCE_GROUP_DUMMY_B, system_a, positions_a, a_map),
        ("lambda=1 (B)", 1.0, RBFE_FORCE_GROUP_DUMMY_A, system_b, positions_b, b_map),
    ):
        parameters = {name: lam for name in RBFE_LAMBDA_NAMES}
        reference_no_lrc = _without_dispersion_correction(reference_system)

        total_e, total_f = _energy_and_forces(
            hybrid_no_lrc, positions_q, parameters, platform_name=platform_name
        )
        dummy_e, dummy_f = _energy_and_forces(
            hybrid_no_lrc, positions_q, parameters,
            platform_name=platform_name, groups={other_group},
        )
        ref_e, ref_f = _energy_and_forces(
            reference_no_lrc, reference_positions, platform_name=platform_name
        )
        separable_e = total_e - dummy_e

        # LRC 缺口：两边"开 LRC 减关 LRC"的差，就是 hybrid 少掉的那部分炼金 LJ 尾项。
        hybrid_with_lrc, _ = _energy_and_forces(
            bundle.system, positions_q, parameters, platform_name=platform_name
        )
        reference_with_lrc, _ = _energy_and_forces(
            reference_system, reference_positions, platform_name=platform_name
        )
        lrc_gap = (hybrid_with_lrc - total_e) - (reference_with_lrc - ref_e)

        force_deviation = 0.0
        for source_index in range(reference_system.getNumParticles()):
            hybrid_index = index_map[source_index]
            delta = (total_f[hybrid_index] - dummy_f[hybrid_index]) - ref_f[source_index]
            force_deviation = max(force_deviation, float(np.max(np.abs(delta))))

        energy_deviation = abs(separable_e - ref_e)
        ok = (
            energy_deviation <= energy_tolerance_kj_per_mol
            and force_deviation <= force_tolerance_kj_per_mol_nm
        )
        report["endpoints"][label] = {
            "hybrid_total_kJ_per_mol": total_e,
            "separable_dummy_kJ_per_mol": dummy_e,
            "hybrid_minus_dummy_kJ_per_mol": separable_e,
            "reference_kJ_per_mol": ref_e,
            "energy_deviation_kJ_per_mol": energy_deviation,
            "max_force_deviation_kJ_per_mol_nm": force_deviation,
            "lj_lrc_gap_kJ_per_mol": float(lrc_gap),
            "passed": bool(ok),
        }
        report["passed"] = report["passed"] and ok
    return report


def verify_hybrid_forces_finite_difference(
    bundle: HybridSystemBundle,
    positions_hybrid,
    lambda_state_index: int,
    *,
    platform_name: str = "Reference",
    delta_nm: float = 2.0e-5,
    n_samples: int = 12,
    seed: int = 20260903,
    relative_tolerance: float = 5.0e-4,
    absolute_tolerance_kj_per_mol_nm: float = 1.0e-2,
) -> dict:
    """计划 §8 的 R1 验收：**有限差分力与解析力一致**。

    对随机抽取的 (原子, 分量) 做中心差分 −dU/dx，与解析力逐个比。抽样是**确定性**
    的（固定 seed），同一个体系跑两次抽到同一批点，否则"通过"不可复现。

    这条测的是 builder 写的表达式与它自己声称的势是否自洽——尤其是三个 custom
    LJ 力那几个手写的 softcore 分母，以及成键插值项。
    """
    from openmm import unit

    positions = np.asarray(
        [[v.x, v.y, v.z] if hasattr(v, "x") else list(v) for v in positions_hybrid], dtype=float
    )
    parameters = bundle.schedule.state(lambda_state_index)
    _, analytic = _energy_and_forces(
        bundle.system, positions * unit.nanometer, parameters, platform_name=platform_name
    )

    rng = np.random.default_rng(seed)
    n_particles = positions.shape[0]
    picks = [
        (int(rng.integers(0, n_particles)), int(rng.integers(0, 3)))
        for _ in range(int(n_samples))
    ]

    rows = []
    worst = 0.0
    for atom, axis in picks:
        shifted_plus = positions.copy()
        shifted_minus = positions.copy()
        shifted_plus[atom, axis] += delta_nm
        shifted_minus[atom, axis] -= delta_nm
        e_plus, _ = _energy_and_forces(
            bundle.system, shifted_plus * unit.nanometer, parameters, platform_name=platform_name
        )
        e_minus, _ = _energy_and_forces(
            bundle.system, shifted_minus * unit.nanometer, parameters, platform_name=platform_name
        )
        numeric = -(e_plus - e_minus) / (2.0 * delta_nm)
        exact = float(analytic[atom, axis])
        deviation = abs(numeric - exact)
        scale = max(abs(exact), abs(numeric), 1.0)
        relative = deviation / scale
        worst = max(worst, relative if deviation > absolute_tolerance_kj_per_mol_nm else 0.0)
        rows.append({
            "atom": atom, "axis": axis,
            "analytic_kJ_per_mol_nm": exact,
            "finite_difference_kJ_per_mol_nm": numeric,
            "abs_deviation": deviation,
            "relative_deviation": relative,
            "passed": bool(
                deviation <= absolute_tolerance_kj_per_mol_nm
                or relative <= relative_tolerance
            ),
        })

    return {
        "lambda_state_index": int(lambda_state_index),
        "lambda_values": parameters,
        "delta_nm": float(delta_nm),
        "n_samples": len(rows),
        "worst_relative_deviation": float(worst),
        "samples": rows,
        "passed": all(row["passed"] for row in rows),
    }


def hybrid_charge_ledger(bundle: HybridSystemBundle, *, tolerance_e: float = 1e-9) -> dict:
    """计划 §5.3：「**核对每个 λ 的有效总电荷**。即使 A、B 净电荷相同，也不能因
    不同电荷分段切换而忽略中间态的电荷变化。」

    直接读**建好的** NonbondedForce 的粒子电荷与 offset，按 λ 表逐态求和——
    不是把 builder 的意图重算一遍，而是核对它真正写进去的东西。
    """
    import openmm
    from openmm import unit

    nb = _nb_force_of(bundle.system, "hybrid 体系")
    alchemical = set(bundle.layout.alchemical)

    base = {}
    for index in sorted(alchemical):
        charge, _sigma, _epsilon = nb.getParticleParameters(index)
        base[index] = _quantity_value(charge)

    offsets = {}
    for i in range(nb.getNumParticleParameterOffsets()):
        name, index, charge_scale, _sigma_scale, _eps_scale = nb.getParticleParameterOffset(i)
        if name != LAMBDA_CHARGE:
            continue
        offsets[int(index)] = offsets.get(int(index), 0.0) + _quantity_value(charge_scale)

    stray = sorted(set(offsets) - alchemical)
    if stray:
        raise RBFEHybridBuildError(
            f"非炼金原子 {stray} 上挂着 {LAMBDA_CHARGE} 的电荷 offset——环境电荷不该随 λ 变"
        )

    rows = []
    for index in range(bundle.schedule.n_states):
        lam = bundle.schedule.state(index)[LAMBDA_CHARGE]
        total = sum(base[i] + lam * offsets.get(i, 0.0) for i in sorted(alchemical))
        rows.append({"state": index, LAMBDA_CHARGE: lam, "alchemical_net_charge_e": total})

    endpoints = (rows[0]["alchemical_net_charge_e"], rows[-1]["alchemical_net_charge_e"])
    max_excursion = max(
        abs(row["alchemical_net_charge_e"] - endpoints[0]) for row in rows
    )
    return {
        "per_state": rows,
        "net_charge_A_e": endpoints[0],
        "net_charge_B_e": endpoints[1],
        "endpoints_match": bool(abs(endpoints[0] - endpoints[1]) <= tolerance_e),
        "max_intermediate_excursion_e": float(max_excursion),
        "constant_across_lambda": bool(max_excursion <= tolerance_e),
    }


# ---------------------------------------------------------------------------
# R2：对真实 hybrid Hamiltonian 做 MBAR
# ---------------------------------------------------------------------------
#
# 计划 §6 明确禁止复用 `TraditionalMBARAnalyzer.compute_u_kn`——它按**单配体去耦**
# 和 LRC 假设重建评估系统（PME self correction、ligand_charge_square_sum、
# co-ion、Boresch……），那套假设对 hybrid Hamiltonian 全都不成立。
#
# 但同一句话的后半段也要照做：计划 §6 说**独立的 MBAR 数值求解部分**可以复用。
# 所以这里的分工是：
#
#   u_kn        —— RBFE 自己算，直接在 hybrid System 上换 λ 求能量（本节）
#   MBAR 求解   —— 复用 `ibs_engine.TraditionalMBARAnalyzer.solve()`
#                  （去相关子采样、overlap 诊断、多套 solver protocol 兜底，
#                    都是那边踩出来的，重写必错）
#
# u_kn 这边比 ABFE 简单得多，而且**结构上不可能算错评估体系**：hybrid System 只有
# 一个，换状态就是 `context.setParameter(λ)`，不重建任何东西。ABFE 那边之所以复杂，
# 正是因为它要为每个 λ 重新造一个"等效的去耦体系"。

#: 1 bar·nm³ 换算成 kJ/mol。
#: 1e5 Pa × 1e-27 m³ = 1e-22 J；× N_A(6.02214076e23 /mol) = 60.2214076 J/mol。
_KJ_PER_MOL_PER_BAR_NM3 = 0.0602214076

#: 气体常数 R，kJ/(mol·K)。与 `ibs_engine.TraditionalMBARAnalyzer` 用的是同一个值——
#: 两边必须一致，否则 u_kn 的 β 和求解器的 kT 对不上，ΔG 会差一个常数因子。
_R_KJ_PER_MOL_K = 0.008314462618


def compute_hybrid_u_kn(
    bundle: HybridSystemBundle,
    samples,
    *,
    temperature_kelvin: float,
    pressure_bar: Optional[float] = None,
    platform_name: str = "Reference",
) -> dict:
    """在**真实 hybrid Hamiltonian** 上求约化能量矩阵 u_kn。

    `samples` 是 `free_energy_engine.run_independent_windows` 的产物（鸭子类型：
    需要 `positions_by_state` / `box_vectors_by_state` / `state_parameters`）。

    ## 样本归属靠下标，不靠文件名

    计划 §7：「能量矩阵记录维度、状态顺序、样本状态索引、是否约化和是否包含 NPT
    所需项；**不按文件名猜**。」这里 `samples.positions_by_state[k]` 就是状态 k 的
    帧，`n_k[k]` 是它的帧数，返回值把这些一并带出去。

    进函数第一件事是核对 `samples.state_parameters` 与 `bundle.schedule` **逐位相同**
    ——样本是在哪个 λ 上采的、u_kn 又要在哪些 λ 上评估，这两件事必须由同一张表决定。
    对不上就报错，不做任何"看起来能对上"的迁就。

    ## NPT

    `pressure_bar` 给了就在约化势里加 βpV（V 取**每一帧自己的**盒体积）。
    不给就是 NVT。返回值里 `includes_pV` 如实记录，不让下游去猜。
    """
    import openmm
    from openmm import unit

    schedule_states = [bundle.schedule.state(i) for i in range(bundle.schedule.n_states)]
    sample_states = [dict(p) for p in samples.state_parameters]
    if len(sample_states) != len(schedule_states):
        raise RBFEHybridBuildError(
            f"样本有 {len(sample_states)} 个状态，λ 表有 {len(schedule_states)} 个。"
            "样本必须是在这张 λ 表上采的。"
        )
    for index, (sampled, expected) in enumerate(zip(sample_states, schedule_states)):
        if set(sampled) != set(expected) or any(
            abs(float(sampled[name]) - float(expected[name])) > 1e-12 for name in expected
        ):
            raise RBFEHybridBuildError(
                f"第 {index} 个状态的 λ 与 λ 表不一致：样本 {sampled} vs λ 表 {expected}。"
                "样本采样时用的 λ 与评估用的 λ 必须来自同一张表。"
            )

    sample_pressure = getattr(samples, "pressure_bar", None)
    if (sample_pressure is None) != (pressure_bar is None):
        raise RBFEHybridBuildError(
            f"系综不一致：样本是 pressure_bar={sample_pressure} 采的，"
            f"却要求按 pressure_bar={pressure_bar} 约化。"
            "NPT 样本按 NVT 约化会漏掉 βpV，反之会凭空多一项。"
        )

    beta = 1.0 / (_R_KJ_PER_MOL_K * float(temperature_kelvin))
    n_states = len(schedule_states)
    n_k = np.asarray([len(frames) for frames in samples.positions_by_state], dtype=int)
    if np.any(n_k <= 0):
        raise RBFEHybridBuildError(f"有状态一帧都没有：n_k={n_k.tolist()}")
    total = int(n_k.sum())

    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(bundle.system, integrator, platform)

    u_kn = np.zeros((n_states, total), dtype=float)
    sample_state_index = np.zeros(total, dtype=int)
    column = 0
    try:
        for k in range(n_states):
            frames = samples.positions_by_state[k]
            boxes = samples.box_vectors_by_state[k]
            for frame_index in range(len(frames)):
                box = np.asarray(boxes[frame_index], dtype=float)
                context.setPeriodicBoxVectors(*[openmm.Vec3(*row) * unit.nanometer for row in box])
                context.setPositions(np.asarray(frames[frame_index], dtype=float) * unit.nanometer)
                pv_term = 0.0
                if pressure_bar is not None:
                    volume = abs(float(np.linalg.det(box)))
                    pv_term = float(pressure_bar) * volume * _KJ_PER_MOL_PER_BAR_NM3
                for evaluated in range(n_states):
                    for name, value in schedule_states[evaluated].items():
                        context.setParameter(name, float(value))
                    energy = context.getState(getEnergy=True).getPotentialEnergy()
                    energy = energy.value_in_unit(unit.kilojoule_per_mole)
                    if not math.isfinite(energy):
                        raise RBFEHybridBuildError(
                            f"状态 {k} 第 {frame_index} 帧在 λ 态 {evaluated} 上能量非有限：{energy}"
                        )
                    u_kn[evaluated, column] = beta * (energy + pv_term)
                sample_state_index[column] = k
                column += 1
    finally:
        del context, integrator

    return {
        "u_kn": u_kn,
        "n_k": n_k,
        "sample_state_index": sample_state_index,
        "n_states": int(n_states),
        "n_samples": total,
        "reduced": True,
        "includes_pV": pressure_bar is not None,
        "temperature_kelvin": float(temperature_kelvin),
        "pressure_bar": pressure_bar,
        "beta_mol_per_kJ": float(beta),
        "state_order": [dict(s) for s in schedule_states],
        "hybrid_fingerprint": bundle.fingerprint(),
        "evaluated_on": "hybrid_system_via_setParameter__no_system_rebuild",
    }


def analyze_leg(
    bundle: HybridSystemBundle,
    samples,
    *,
    phase: str,
    edge_id: str,
    ligand_a_name: str,
    ligand_b_name: str,
    temperature_kelvin: float,
    pressure_bar: Optional[float] = None,
    energy_unit: str = KJ_PER_MOL,
    platform_name: str = "Reference",
    decorrelate: bool = True,
) -> tuple:
    """R2：由样本算出这条腿的 ΔG(A→B)。返回 `(LegResult, 诊断字典)`。

    ## 方向约定的落点（计划 §3）

    `TraditionalMBARAnalyzer.solve()` 返回的 `delta_G` 是
    `(f[-1] − f[0]) · kT`，即**最后一个状态减第一个状态**。本 builder 的 λ 表被
    `HybridLambdaSchedule` 强制成端点 0→1，而 λ=0 对应 A、λ=1 对应 B，所以

        delta_G == G(B) − G(A) == ΔG(A→B)

    正是 `LegResult.delta_g` 要求的含义。**这条链不能靠记忆维持**——
    `test_leg_result_direction_matches_the_plan` 用一个符号已知的构造把它钉死。

    ## 与计划里那个签名的差别

    计划 §4 草案写的是 `analyze_leg(leg: PreparedLeg, artifacts: SamplingArtifacts)`。
    `PreparedLeg` / `SamplingArtifacts` 那两层还没有（前者要 `prepare_edge`，
    后者是落盘轨迹的路径），所以这里直接吃 `HybridSystemBundle` + 内存样本。
    接口形状变了，职责没变。
    """
    if phase not in RBFE_PHASES:
        raise ValueError(f"phase 必须是 {RBFE_PHASES} 之一：收到 {phase!r}")

    matrix = compute_hybrid_u_kn(
        bundle,
        samples,
        temperature_kelvin=temperature_kelvin,
        pressure_bar=pressure_bar,
        platform_name=platform_name,
    )

    # 🔑 复用 ABFE 的 MBAR 数值求解（计划 §6 允许的那一部分）。**惰性 import**：
    # `ibs_engine` 是两万多行、且会拉进 openmm/pymbar，放模块顶部会让 R0/R1a
    # 那两层「不 import openmm」的性质失效。这里只用它的 solve()，
    # 一行 ABFE 代码都不改。
    import ibs_engine

    analyzer = ibs_engine.TraditionalMBARAnalyzer(temperature=float(temperature_kelvin))
    analyzer._last_n_k = matrix["n_k"]
    solved = analyzer.solve(matrix["u_kn"], decorrelate=decorrelate)

    delta_g_kj = float(solved["delta_G"])
    stderr_kj = float(solved.get("error", float("nan")))
    if not math.isfinite(stderr_kj):
        raise RBFEHybridBuildError(
            "MBAR 没有给出不确定度（error 为 nan）。没有误差棒的 ΔG 不能进 LegResult——"
            "下游的 ΔΔG 误差传播会静默变成 nan，而 nan 一路传到最终报告里很难被发现。"
        )

    delta_g = convert_energy(delta_g_kj, KJ_PER_MOL, energy_unit)
    stderr = convert_energy(stderr_kj, KJ_PER_MOL, energy_unit)

    effective = solved.get("diagnostics", {}).get("effective_sample_number")
    if isinstance(effective, (list, tuple)) and effective:
        n_effective = int(min(float(v) for v in effective))
    else:
        n_effective = int(solved.get("n_frames", matrix["n_samples"]))

    fingerprint_payload = {
        "hybrid_fingerprint": matrix["hybrid_fingerprint"],
        "n_k": matrix["n_k"].tolist(),
        "reduced": matrix["reduced"],
        "includes_pV": matrix["includes_pV"],
        "temperature_kelvin": matrix["temperature_kelvin"],
        "pressure_bar": matrix["pressure_bar"],
        "state_order": matrix["state_order"],
        "sampler": dict(getattr(samples, "provenance", {}) or {}),
        "mbar_method": solved.get("method"),
        "decorrelate": bool(decorrelate),
    }
    artifacts_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()

    result = LegResult(
        phase=phase,
        edge_id=edge_id,
        ligand_a_name=ligand_a_name,
        ligand_b_name=ligand_b_name,
        delta_g=delta_g,
        stderr=stderr,
        energy_unit=energy_unit,
        uncertainty_method=str(solved.get("method", "mbar")),
        n_effective_samples=n_effective,
        quality_gate_passed=bool(solved.get("converged", False)),
        artifacts_fingerprint=artifacts_fingerprint,
    )
    diagnostics = {
        "u_kn_shape": list(matrix["u_kn"].shape),
        "n_k": matrix["n_k"].tolist(),
        "includes_pV": matrix["includes_pV"],
        "min_overlap": solved.get("min_overlap"),
        "converged": solved.get("converged"),
        "mbar": {key: value for key, value in solved.items() if key != "diagnostics"},
        "mbar_diagnostics": solved.get("diagnostics", {}),
        "sampler_provenance": dict(getattr(samples, "provenance", {}) or {}),
    }
    return result, diagnostics


# === 未实现区（R1 起）=====================================================
#
# 计划 §4.1：「不提供『尚未实现但返回成功』的占位 sampler；接口准备完成与可运行
# 科学计算是两个验收阶段。」所以下面这些**抛错**，不返回假数据。
#
# ==========================================================================


def prepare_edge(spec: EdgeSpec):  # pragma: no cover - 未实现
    """从 `EdgeSpec` 的输入路径建出两个端点体系并冻结映射。**尚未实现。**

    缺的是「读参数化输入 → 建 System → 同一个盒子里溶剂化 A 和 B」这一段，
    它需要一份**可跑的**配体 B（改基团必然要重新给部分电荷与成键参数，
    §5.1 不允许悄悄重参数化），是 R3 的硬前提。

    映射与 hybrid builder 都已经可用，可以直接用：
        `map_atoms` / `validate_mapping` / `build_hybrid_system`
    """
    raise NotImplementedError(
        "prepare_edge 尚未实现：它要从 EdgeSpec 的输入路径建出 A、B 两个端点体系，"
        "前提是有一份可跑的配体 B（含部分电荷与成键参数）。"
        "映射用 map_atoms、hybrid 体系用 build_hybrid_system，两者都已可用。"
        "见 docs/design/PLAN_rbfe_interface_and_implementation.md §0。"
    )


def build_hybrid_leg(prepared, phase: str):  # pragma: no cover - 等 prepare_edge
    """按 `PreparedEdge` 建一条腿的 hybrid 体系。**这一层还没有**。

    ⚠ 别被这个 `NotImplementedError` 误导成「hybrid builder 没做」——做了，
    是 `build_hybrid_system()`（R1b，已实现并通过端点等价 + 有限差分力验收）。
    缺的只是它上面这层薄封装，因为 `PreparedEdge` 要等 `prepare_edge`
    （从 `EdgeSpec` 的输入路径建出两个 System）才有定义。

    现在就能用的调用方式：

        bundle = rbfe_core.build_hybrid_system(
            system_a, ligand_indices_a, system_b, ligand_indices_b, mapping, schedule
        )
    """
    raise NotImplementedError(
        "build_hybrid_leg 需要 prepare_edge 产出的 PreparedEdge，那一层尚未实现。"
        "hybrid builder 本身已经可用：直接调 rbfe_core.build_hybrid_system("
        "system_a, ligand_indices_a, system_b, ligand_indices_b, mapping, schedule)。"
    )



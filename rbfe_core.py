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

本模块不 import `rbfe_pipeline` / `free_energy_engine` / `abfe_*` / `ibs_engine`。

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

import math
import re
from dataclasses import dataclass, field
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


def validate_edge(spec: EdgeSpec) -> ValidationReport:
    """R0 验证：把首版范围外的输入在**创建任何 Context 之前**拒掉（计划 §2）。

    首版明令拒绝的变换（计划 §2 逐条）：净电荷变化、带电配体、膜体系、质子化／
    互变异构改变。环断裂／闭合、环尺寸变化、手性反转、映射元素改变、共价配体
    这几类需要原子映射才能判定，属于 R1——本函数把它们列进 `unchecked`，
    **不假装已经查过**。
    """
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

    # 🔑 诚实记录本阶段查不了的东西。这几项全都需要原子映射（R1），
    # R0 没有映射，所以既不能通过也不能拒绝，只能声明"没查"。
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


# === 未实现区（R1 起）=====================================================
#
# 计划 §4.1：「不提供『尚未实现但返回成功』的占位 sampler；接口准备完成与可运行
# 科学计算是两个验收阶段。」所以下面这些**抛错**，不返回假数据。
#
# ==========================================================================


def prepare_edge(spec: EdgeSpec):  # pragma: no cover - R1
    """R1：原子映射 + 端点建系。尚未实现。"""
    raise NotImplementedError(
        "prepare_edge 属于 R1（映射与受限 hybrid builder），尚未实现。"
        "当前只有 R0：validate_edge / combine_rbfe。"
        "见 docs/design/PLAN_rbfe_interface_and_implementation.md §8。"
    )


def build_hybrid_leg(prepared, phase: str):  # pragma: no cover - R1
    """R1：构建 hybrid 拓扑与 Hamiltonian。尚未实现。"""
    raise NotImplementedError(
        "build_hybrid_leg 属于 R1，尚未实现。计划 §5.3 要求先明确 common core / "
        "A-only / B-only 原子的 charge、LJ、bonded、exception、约束与 dummy 处理，"
        "并验证每个 λ 的有效总电荷——这些都还没有做。"
    )


def analyze_leg(prepared, artifacts):  # pragma: no cover - R2
    """R2：对真实 hybrid Hamiltonian 做 MBAR。尚未实现。"""
    raise NotImplementedError(
        "analyze_leg 属于 R2，尚未实现。计划 §6 明确：不能复用 "
        "TraditionalMBARAnalyzer.compute_u_kn——它按单配体去耦和 LRC 假设重建评估"
        "系统，RBFE 必须对实际 hybrid Hamiltonian 评估。"
    )

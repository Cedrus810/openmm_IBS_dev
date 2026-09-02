#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RBFE 编排层：运行目录与身份、两腿、独立重复、边网络、续跑校验。

设计依据：`docs/design/PLAN_rbfe_interface_and_implementation.md` §4 / §7 / §8。

## 职责边界（§4 表格，逐字）

本模块负责：prepare/run/analyze 编排、两腿、独立重复、边集合／网络、运行状态和续跑校验。
本模块**不负责**：自行构建相互作用公式、重写交换算法。

因此：hybrid Hamiltonian 归 `rbfe_core`（R1），采样推进归 `free_energy_engine`，
本模块只决定「什么时候、按什么身份、把哪份数据交给谁」。

## 当前实现进度

已经能用（不依赖 R1，契约已经钉死）：

* 运行目录布局与 manifest 落盘（§7）
* **边身份指纹**与续跑复用规则（§7：禁止跨 A/B 方向、映射、力场、后端或 λ 路径直接追加）
* 两腿编排与两腿间的身份校验
* 独立重复的聚合：重复间方差、跨重复一致性
* 边网络：连通性、闭合环、环闭合残差与不确定度传播
* ABFE 锚点换算（把 ΔΔG 网络钉到绝对值），含锚点误差传播

**尚未实现**（R1/R2，本模块只负责调用，不负责实现）：原子映射、hybrid builder、
真实采样与 MBAR。`prepare_edge` / `run_leg` / `analyze_leg` 走到那一步会由
`rbfe_core` 抛 `NotImplementedError`——**不提供「尚未实现但返回成功」的占位**
（计划 §4.1）。

## 依赖方向（不得违反）

    runrbfe.py -> rbfe_pipeline.py -> rbfe_core.py
                                   -> free_energy_engine.py

本模块不 import `abfe_core` / `abfe_pipeline` / `ibs_engine` / `runabfe`。
ABFE 与 RBFE 是两条独立入口，只共用经过抽象和验证的底层组件（计划 §1）。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import free_energy_engine as fee
import rbfe_core as rc

#: 编排层协议版本。改变运行目录布局、身份指纹构成或重复聚合口径都必须 +1。
RBFE_PIPELINE_PROTOCOL_VERSION = 1

#: 运行目录里的固定文件名（计划 §7）。
MANIFEST_NAME = "edge_manifest.json"
ATOM_MAPPING_NAME = "atom_mapping.json"
ENDPOINT_VALIDATION_NAME = "endpoint_validation.json"
RESULT_NAME = "rbfe_result.json"
LEG_RESULT_NAME = "leg_result.json"


class RBFEPipelineError(RuntimeError):
    """编排层的 fail-closed 错误。"""


class RBFEResumeError(RBFEPipelineError):
    """续跑／复用身份不匹配。**宁可重跑，也不混用两次不同协议的样本。**"""


# ---------------------------------------------------------------------------
# 身份指纹
# ---------------------------------------------------------------------------
#
# 🔑 本仓库在 ABFE 那边为「缓存键里混进了非身份的东西」付过很大代价：键里一旦掺入
# 执行历史（代码哈希、时间戳、已跑步数），任何无关改动都会让 resume 被迫重跑 GPU。
# 这里从一开始就划清楚：**指纹只放身份，不放执行历史。**


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256_of(payload: Any) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def edge_identity(
    spec: rc.EdgeSpec,
    *,
    atom_mapping_hash: Optional[str] = None,
    hybrid_builder_version: Optional[str] = None,
) -> dict:
    """一条边的**身份**——两次运行只要这里不同，产物就不可互换。

    计划 §7 点名了必须进身份的东西：A/B 化学及输入身份、映射和参数 hash、
    hybrid builder 版本、λ 路径、方向、单位。

    刻意**不放进来**的（它们是执行历史，不是身份）：
      * 代码文件的 sha256——算法变了应该由协议版本号声明，不是靠代码哈希兜底；
      * 已完成步数、耗时、时间戳、主机名；
      * 平台与后端——它们进的是**采样产物**的指纹（见 `leg_identity`），
        换平台不该让整条边的准备工作作废。
    """
    return {
        "rbfe_core_protocol_version": int(spec.protocol_version),
        "rbfe_pipeline_protocol_version": RBFE_PIPELINE_PROTOCOL_VERSION,
        "edge_id": spec.edge_id,
        "direction": rc.RBFE_DIRECTION,
        "energy_unit": spec.energy_unit,
        "ligand_A": spec.ligand_a.identity(),
        "ligand_B": spec.ligand_b.identity(),
        "receptor_sha256": spec.environment.receptor_sha256,
        "force_field": spec.environment.force_field,
        "water_model": spec.environment.water_model,
        "ion_model": spec.environment.ion_model,
        "is_membrane": bool(spec.environment.is_membrane),
        "temperature_kelvin": float(spec.protocol.temperature_kelvin),
        "pressure_bar": spec.protocol.pressure_bar,
        "n_lambda_states": int(spec.protocol.n_lambda_states),
        "lambda_schedule_name": spec.protocol.lambda_schedule_name,
        "atom_mapping_sha256": atom_mapping_hash,
        "hybrid_builder_version": hybrid_builder_version,
    }


def edge_fingerprint(spec: rc.EdgeSpec, **kwargs: Any) -> str:
    return _sha256_of(edge_identity(spec, **kwargs))


def leg_identity(
    spec: rc.EdgeSpec,
    phase: str,
    *,
    backend: Optional[fee.BackendResolution] = None,
    **kwargs: Any,
) -> dict:
    """一条腿的采样身份 = 边身份 + 相 + 采样预算 + 实际后端。

    后端进**腿**的身份而不是边的身份：换后端会让已采的样本不可续接（计划 §4.3
    「禁止跨后端直接加载动态 checkpoint 或向旧后端的未完成 DCD 追加帧」），
    但不该让映射和建系那一步作废。
    """
    if phase not in rc.RBFE_PHASES:
        raise ValueError(f"phase 必须是 {rc.RBFE_PHASES} 之一：{phase!r}")
    payload = {
        "edge": edge_identity(spec, **kwargs),
        "phase": phase,
        "n_steps_per_state": int(spec.protocol.n_steps_per_state),
        "seed": int(spec.protocol.seed),
    }
    if backend is not None:
        payload["backend"] = backend.resolved
        payload["exchange_scheme"] = backend.exchange_scheme
        payload["adapter_protocol_version"] = int(backend.adapter_protocol_version)
    return payload


def leg_fingerprint(spec: rc.EdgeSpec, phase: str, **kwargs: Any) -> str:
    return _sha256_of(leg_identity(spec, phase, **kwargs))


# ---------------------------------------------------------------------------
# 运行目录
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunLayout:
    """一次重复的目录布局（计划 §7）。"""

    root: Path
    repeat_index: int

    @property
    def repeat_dir(self) -> Path:
        return self.root / f"repeat_{self.repeat_index:02d}"

    @property
    def manifest_path(self) -> Path:
        return self.repeat_dir / MANIFEST_NAME

    @property
    def result_path(self) -> Path:
        return self.repeat_dir / RESULT_NAME

    def leg_dir(self, phase: str) -> Path:
        if phase not in rc.RBFE_PHASES:
            raise ValueError(f"未知 phase：{phase!r}")
        return self.repeat_dir / phase

    def leg_result_path(self, phase: str) -> Path:
        return self.leg_dir(phase) / LEG_RESULT_NAME


def _atomic_write_json(path: Path, payload: Any) -> None:
    """原子写。

    manifest 与结果会被续跑和外部审计读取；进程在 open() 与 dump() 之间被打断，
    绝不能留下一份「看着合法但内容截断」的文档。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def prepare_run_directory(
    spec: rc.EdgeSpec,
    repeat_index: int = 1,
    *,
    validate: bool = True,
    **identity_kwargs: Any,
) -> RunLayout:
    """建立一次重复的运行目录并写下 manifest。

    `validate=True` 时先跑 R0 验证并 fail-closed——**在建任何目录之前**，
    免得留下一堆注定跑不成的空目录。
    """
    if validate:
        rc.validate_edge(spec).raise_if_failed()

    layout = RunLayout(root=Path(spec.output_dir), repeat_index=int(repeat_index))
    manifest = dict(spec.manifest())
    manifest["repeat_index"] = int(repeat_index)
    manifest["rbfe_pipeline_protocol_version"] = RBFE_PIPELINE_PROTOCOL_VERSION
    manifest["edge_fingerprint"] = edge_fingerprint(spec, **identity_kwargs)
    manifest["edge_identity"] = edge_identity(spec, **identity_kwargs)

    if layout.manifest_path.exists():
        assert_reusable(layout, spec, **identity_kwargs)
    _atomic_write_json(layout.manifest_path, manifest)
    for phase in rc.RBFE_PHASES:
        layout.leg_dir(phase).mkdir(parents=True, exist_ok=True)
    return layout


def assert_reusable(
    layout: RunLayout, spec: rc.EdgeSpec, **identity_kwargs: Any
) -> dict:
    """已有运行目录能否复用。不匹配就抛 `RBFEResumeError`。

    计划 §7：**禁止跨 A/B 方向、映射、力场、后端或 λ 路径直接追加。**
    这里做的是"拒绝"，不是"清理"——不自动删除别人的产物，让用户自己决定
    是换输出目录还是确认覆盖。
    """
    try:
        existing = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RBFEResumeError(
            f"{layout.manifest_path} 存在但读不出来（{exc}）——"
            "拒绝在一个身份不明的目录里继续，请换 --output 或手工清理"
        ) from exc

    old = existing.get("edge_identity")
    if not old:
        raise RBFEResumeError(
            f"{layout.manifest_path} 里没有 edge_identity——"
            "它不是本协议写出来的，拒绝复用"
        )

    new = edge_identity(spec, **identity_kwargs)
    diffs = [
        f"{k}: 已有={old.get(k)!r} 现在={new[k]!r}"
        for k in sorted(set(old) | set(new))
        if old.get(k) != new.get(k)
    ]
    if diffs:
        raise RBFEResumeError(
            "运行目录的边身份与当前配置不一致，拒绝复用：\n  - " + "\n  - ".join(diffs)
        )
    return existing


# ---------------------------------------------------------------------------
# 两腿编排
# ---------------------------------------------------------------------------


def run_leg(
    spec: rc.EdgeSpec,
    phase: str,
    layout: RunLayout,
    *,
    backend: Optional[fee.BackendResolution] = None,
    **identity_kwargs: Any,
):  # pragma: no cover - 依赖 R1/R2
    """跑一条腿。**R1/R2 未实现，这里会抛 NotImplementedError。**

    编排层已经就位：身份指纹算得出、目录建得好、后端决议拿得到；缺的是
    `rbfe_core.build_hybrid_leg`（R1）——那一步没有，就没有可采样的 Hamiltonian。
    """
    if phase not in rc.RBFE_PHASES:
        raise ValueError(f"未知 phase：{phase!r}")
    if backend is None:
        backend = fee.resolve_remd_backend()
    prepared = rc.prepare_edge(spec)          # R1：抛 NotImplementedError
    leg = rc.build_hybrid_leg(prepared, phase)  # R1
    request = fee.SamplingRequest(            # pragma: no cover - 到不了这里
        system=leg.system,
        topology=leg.topology,
        states=leg.states,
        initial_positions=leg.initial_positions,
        initial_box_vectors=leg.initial_box_vectors,
        temperature_kelvin=spec.protocol.temperature_kelvin,
        pressure_bar=spec.protocol.pressure_bar,
        total_md_steps=spec.protocol.n_steps_per_state,
        output_dir=str(layout.leg_dir(phase)),
        stage_name=f"{spec.edge_id}_{phase}",
        caller_protocol_fingerprint=leg_fingerprint(
            spec, phase, backend=backend, **identity_kwargs
        ),
    )
    artifacts = fee.run_sampling(request, leg.sampler, backend)
    return rc.analyze_leg(leg, artifacts)     # R2


def combine_legs(
    spec: rc.EdgeSpec,
    complex_result: rc.LegResult,
    solvent_result: rc.LegResult,
    layout: Optional[RunLayout] = None,
    *,
    covariance: float = 0.0,
) -> rc.EdgeResult:
    """两腿 -> ΔΔG，并（给了 layout 时）落盘。

    符号与误差传播全部委托给 `rbfe_core.combine_rbfe`——编排层**不自己算科学量**。
    这里只多做一件编排层该做的事：确认两腿确实来自这条 spec，而不只是彼此自洽。
    """
    for leg in (complex_result, solvent_result):
        if leg.edge_id != spec.edge_id:
            raise RBFEPipelineError(
                f"{leg.phase} 腿的 edge_id={leg.edge_id!r} 与 spec 的 "
                f"{spec.edge_id!r} 不一致"
            )
        if leg.ligand_a_name != spec.ligand_a.name or leg.ligand_b_name != spec.ligand_b.name:
            raise RBFEPipelineError(
                f"{leg.phase} 腿的配体身份与 spec 不一致："
                f"({leg.ligand_a_name}, {leg.ligand_b_name}) vs "
                f"({spec.ligand_a.name}, {spec.ligand_b.name})"
            )

    result = rc.combine_rbfe(complex_result, solvent_result, covariance=covariance)
    if layout is not None:
        payload = result.to_dict()
        payload["repeat_index"] = layout.repeat_index
        payload["rbfe_pipeline_protocol_version"] = RBFE_PIPELINE_PROTOCOL_VERSION
        _atomic_write_json(layout.result_path, payload)
    return result


# ---------------------------------------------------------------------------
# 独立重复
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepeatAggregate:
    """多次独立重复的聚合（计划 §7：汇总保留重复间方差）。"""

    edge_id: str
    ligand_a_name: str
    ligand_b_name: str
    energy_unit: str
    n_repeats: int
    mean_ddg: float
    #: 重复间标准差（样本标准差，n-1）。n=1 时为 None——**不是 0**。
    between_repeat_sd: Optional[float]
    #: 均值的标准误：重复间 sd / sqrt(n)。n=1 时退回单次重复自己的传播误差。
    stderr_of_mean: float
    #: 各次重复自己报告的传播误差的均方根，用来和 between_repeat_sd 对照。
    mean_reported_stderr: float
    per_repeat: tuple
    qualified: bool
    qualification_reasons: tuple

    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id,
            "direction": rc.RBFE_DIRECTION,
            "ligand_A": self.ligand_a_name,
            "ligand_B": self.ligand_b_name,
            "energy_unit": self.energy_unit,
            "n_repeats": self.n_repeats,
            "ddG_bind_B_minus_A": self.mean_ddg,
            "stderr_of_mean": self.stderr_of_mean,
            "between_repeat_sd": self.between_repeat_sd,
            "mean_reported_stderr": self.mean_reported_stderr,
            "per_repeat_ddG": list(self.per_repeat),
            "qualified": self.qualified,
            "qualification_reasons": list(self.qualification_reasons),
            "rbfe_pipeline_protocol_version": RBFE_PIPELINE_PROTOCOL_VERSION,
        }


def aggregate_repeats(results: Sequence[rc.EdgeResult]) -> RepeatAggregate:
    """聚合独立重复。

    🔑 **重复间散布才是真实不确定度的下限。** 单次重复内部的传播误差经常乐观得多
    （它只反映该次采样的统计涨落，抓不到初始构型／随机源带来的系统性差异）。
    所以这里两个数都留下来，让它们能被并排比较——只报一个的话，读的人无从判断
    单次误差是不是被低估了。
    """
    results = list(results)
    if not results:
        raise RBFEPipelineError("没有可聚合的重复结果")

    first = results[0]
    for r in results[1:]:
        for attr, label in (
            ("edge_id", "edge_id"),
            ("ligand_a_name", "ligand_A"),
            ("ligand_b_name", "ligand_B"),
            ("energy_unit", "energy_unit"),
            ("direction", "direction"),
        ):
            if getattr(r, attr) != getattr(first, attr):
                raise RBFEPipelineError(
                    f"重复之间的 {label} 不一致（{getattr(first, attr)!r} vs "
                    f"{getattr(r, attr)!r}）——拒绝聚合不同边的结果"
                )

    values = [float(r.ddg_bind) for r in results]
    n = len(values)
    mean = sum(values) / n

    if n >= 2:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        sd = math.sqrt(variance)
        stderr = sd / math.sqrt(n)
    else:
        # n=1：重复间散布**无法估计**。返回 None 而不是 0——0 会被读成
        # "重复之间完全一致"，那是一个只跑了一次的运行绝对给不出的结论。
        sd = None
        stderr = float(results[0].ddg_stderr)

    reported = [float(r.ddg_stderr) for r in results]
    rms_reported = math.sqrt(sum(s * s for s in reported) / n)

    reasons: list[str] = []
    for i, r in enumerate(results, start=1):
        if not r.qualified:
            reasons.append(f"repeat_{i:02d}: {'; '.join(r.qualification_reasons)}")
    if n == 1:
        reasons.append("只有 1 次重复，重复间方差无法估计")

    return RepeatAggregate(
        edge_id=first.edge_id,
        ligand_a_name=first.ligand_a_name,
        ligand_b_name=first.ligand_b_name,
        energy_unit=first.energy_unit,
        n_repeats=n,
        mean_ddg=mean,
        between_repeat_sd=sd,
        stderr_of_mean=stderr,
        mean_reported_stderr=rms_reported,
        per_repeat=tuple(values),
        qualified=not reasons,
        qualification_reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# 边网络
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NetworkEdge:
    """网络里的一条有向边 A→B 及其 ΔΔG。"""

    ligand_a: str
    ligand_b: str
    ddg: float
    stderr: float
    energy_unit: str = rc.KJ_PER_MOL

    def __post_init__(self) -> None:
        if self.ligand_a == self.ligand_b:
            raise RBFEPipelineError(f"自环边：{self.ligand_a}→{self.ligand_b}")
        if not math.isfinite(self.ddg) or not math.isfinite(self.stderr):
            raise RBFEPipelineError(f"边 {self.ligand_a}→{self.ligand_b} 的值非有限")
        if self.stderr < 0:
            raise RBFEPipelineError("stderr 不能为负")


@dataclass(frozen=True)
class CycleClosure:
    """一个闭合环的闭合残差。"""

    ligands: tuple
    residual: float
    stderr: float
    energy_unit: str

    @property
    def z_score(self) -> Optional[float]:
        if self.stderr <= 0:
            return None
        return self.residual / self.stderr

    def passes(self, z_threshold: float = 2.0) -> bool:
        z = self.z_score
        if z is None:
            # 误差为 0 时无法做 z 检验。不假装通过——这是"判不了"，不是"通过了"。
            return self.residual == 0.0
        return abs(z) <= z_threshold


@dataclass(frozen=True)
class NetworkReport:
    connected: bool
    components: tuple
    cycles: tuple
    energy_unit: str

    @property
    def all_cycles_close(self) -> bool:
        return all(c.passes() for c in self.cycles)

    def to_dict(self) -> dict:
        return {
            "connected": self.connected,
            "components": [sorted(c) for c in self.components],
            "energy_unit": self.energy_unit,
            "cycles": [
                {
                    "ligands": list(c.ligands),
                    "residual": c.residual,
                    "stderr": c.stderr,
                    "z_score": c.z_score,
                    "passes": c.passes(),
                }
                for c in self.cycles
            ],
            "all_cycles_close": self.all_cycles_close,
            "caveat": (
                "闭合通过不是所有系统误差均已消失的证明——环闭合对"
                "「所有边共享同一个系统性偏差」完全不敏感（计划 §7）。"
            ),
        }


def _connected_components(nodes: set, adjacency: dict) -> list:
    seen: set = set()
    components: list = []
    for start in sorted(nodes):
        if start in seen:
            continue
        stack = [start]
        comp: set = set()
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            stack.extend(n for n in adjacency.get(node, ()) if n not in comp)
        seen |= comp
        components.append(comp)
    return components


def analyze_network(
    edges: Iterable[NetworkEdge], *, max_cycle_length: int = 4
) -> NetworkReport:
    """网络连通性 + 闭合环检查（计划 §7）。

    ΔΔG 是有向量：沿环累加时反向走要取负号。环闭合残差应为 0，其不确定度由
    各边误差平方和传播（把各边视作独立——若它们共享受体构型或参数化，这是乐观的，
    但**低估误差会让检查更严格**，方向是安全的）。

    `max_cycle_length` 默认 4：三角与四元环是实践中真正用来查错的；枚举更长的环
    数量爆炸，而且长环的闭合几乎必然通过（误差累加变大），检不出东西。
    """
    edges = list(edges)
    if not edges:
        raise RBFEPipelineError("网络里没有边")

    units = {e.energy_unit for e in edges}
    if len(units) != 1:
        raise RBFEPipelineError(f"网络里混了多种单位：{sorted(units)}")
    unit = units.pop()

    nodes: set = set()
    adjacency: dict = {}
    directed: dict = {}
    for e in edges:
        nodes.update((e.ligand_a, e.ligand_b))
        adjacency.setdefault(e.ligand_a, set()).add(e.ligand_b)
        adjacency.setdefault(e.ligand_b, set()).add(e.ligand_a)
        key = (e.ligand_a, e.ligand_b)
        if key in directed or (e.ligand_b, e.ligand_a) in directed:
            raise RBFEPipelineError(
                f"重复边 {e.ligand_a}→{e.ligand_b}（同一对配体只应有一条边；"
                "要合并多次测量请先聚合成一条）"
            )
        directed[key] = e

    components = _connected_components(nodes, adjacency)

    def step(u: str, v: str):
        """从 u 走到 v 的 ΔΔG 与误差；反向走取负。"""
        if (u, v) in directed:
            e = directed[(u, v)]
            return e.ddg, e.stderr
        e = directed[(v, u)]
        return -e.ddg, e.stderr

    cycles: list = []
    seen_cycles: set = set()
    ordered = sorted(nodes)
    for start in ordered:
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) > max_cycle_length:
                continue
            for nxt in sorted(adjacency.get(node, ())):
                if nxt == start and len(path) >= 3:
                    canon = frozenset(
                        (min(path[i], path[(i + 1) % len(path)]),
                         max(path[i], path[(i + 1) % len(path)]))
                        for i in range(len(path))
                    )
                    if canon in seen_cycles:
                        continue
                    seen_cycles.add(canon)
                    total = 0.0
                    var = 0.0
                    closed = list(path) + [start]
                    for i in range(len(closed) - 1):
                        d, s = step(closed[i], closed[i + 1])
                        total += d
                        var += s * s
                    cycles.append(
                        CycleClosure(
                            ligands=tuple(path),
                            residual=total,
                            stderr=math.sqrt(var),
                            energy_unit=unit,
                        )
                    )
                elif nxt not in path:
                    stack.append((nxt, path + [nxt]))

    return NetworkReport(
        connected=len(components) == 1,
        components=tuple(frozenset(c) for c in components),
        cycles=tuple(cycles),
        energy_unit=unit,
    )


# ---------------------------------------------------------------------------
# ABFE 锚点
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbsoluteAnchor:
    """把 RBFE 的相对值钉到绝对标度上的锚点（计划 §7）。

    可以是一次 ABFE 计算，也可以是实验值。**只有在受体状态、化学身份、力场、
    温度和自由能定义一致时才允许使用**——这些一致性由调用方负责声明并核对，
    本模块只负责在换算时如实传播锚点自己的误差。
    """

    ligand: str
    delta_g_bind: float
    stderr: float
    energy_unit: str
    source: str
    #: 调用方对"锚点与本网络可比"的显式声明。留空即拒绝使用。
    comparability_statement: str = ""


def absolute_from_anchor(
    report_edges: Iterable[NetworkEdge],
    anchor: AbsoluteAnchor,
) -> dict:
    """由一个绝对锚点推出网络中其它配体的绝对 ΔG_bind。

    沿最短路径累加 ΔΔG，误差按平方和传播并**始终包含锚点自己的误差**——
    离锚点越远误差越大，这一点必须在数字上看得见，不能只报一个中心值。
    """
    edges = list(report_edges)
    if not edges:
        raise RBFEPipelineError("网络里没有边")
    if not anchor.comparability_statement.strip():
        raise RBFEPipelineError(
            f"锚点 {anchor.ligand!r} 没有 comparability_statement——"
            "计划 §7 要求受体状态、化学身份、力场、温度和自由能定义一致时才可换算；"
            "拒绝在没有这条显式声明的情况下把相对值转成绝对值"
        )
    units = {e.energy_unit for e in edges} | {anchor.energy_unit}
    if len(units) != 1:
        raise RBFEPipelineError(f"锚点与网络单位不一致：{sorted(units)}")

    directed: dict = {}
    adjacency: dict = {}
    for e in edges:
        directed[(e.ligand_a, e.ligand_b)] = e
        adjacency.setdefault(e.ligand_a, set()).add(e.ligand_b)
        adjacency.setdefault(e.ligand_b, set()).add(e.ligand_a)

    if anchor.ligand not in adjacency:
        raise RBFEPipelineError(f"锚点配体 {anchor.ligand!r} 不在网络里")

    # 广度优先：跳数最少的路径 = 累加误差最小的路径
    absolute = {
        anchor.ligand: {
            "delta_g_bind": float(anchor.delta_g_bind),
            "stderr": float(anchor.stderr),
            "hops_from_anchor": 0,
        }
    }
    frontier = [anchor.ligand]
    while frontier:
        nxt_frontier = []
        for node in frontier:
            base = absolute[node]
            for neighbour in sorted(adjacency.get(node, ())):
                if neighbour in absolute:
                    continue
                if (node, neighbour) in directed:
                    e = directed[(node, neighbour)]
                    delta = e.ddg
                else:
                    e = directed[(neighbour, node)]
                    delta = -e.ddg
                absolute[neighbour] = {
                    "delta_g_bind": base["delta_g_bind"] + delta,
                    "stderr": math.sqrt(base["stderr"] ** 2 + e.stderr**2),
                    "hops_from_anchor": base["hops_from_anchor"] + 1,
                }
                nxt_frontier.append(neighbour)
        frontier = nxt_frontier

    unreachable = sorted(set(adjacency) - set(absolute))
    return {
        "anchor": {
            "ligand": anchor.ligand,
            "delta_g_bind": anchor.delta_g_bind,
            "stderr": anchor.stderr,
            "source": anchor.source,
            "comparability_statement": anchor.comparability_statement,
        },
        "energy_unit": anchor.energy_unit,
        "absolute": absolute,
        "unreachable_from_anchor": unreachable,
        "caveat": (
            "误差随离锚点的跳数累加；同一网络里不同配体的绝对值精度不同，"
            "不可当作等精度的一组数使用。"
        ),
        "rbfe_pipeline_protocol_version": RBFE_PIPELINE_PROTOCOL_VERSION,
    }

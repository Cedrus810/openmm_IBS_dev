#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RBFE 命令行入口。

设计依据：`docs/design/PLAN_rbfe_interface_and_implementation.md` §4。

## 当前实现进度：只有 `validate`

计划 §8 的 R0 交付物是「schema、验证接口、方向明确的结果汇总、**CLI validate**」。
本文件实现 `validate` 与 `combine`（后者用合成的两腿数据验证符号／单位／误差传播，
不启动任何计算），**不启动 GPU**。

规划中的 `prepare` / `run` / `analyze` 属于 R1-R3，尚未实现——它们在这里注册成
子命令并**明确报错退出**，而不是假装成功。计划 §4.1：「不提供『尚未实现但返回
成功』的占位 sampler；接口准备完成与可运行科学计算是两个验收阶段。」

## 职责边界（§4 表格）

本文件只做：参数解析、配置加载、调用 core/pipeline、用户可见结果与退出码。
**不持有科学算法，也不持有第二份计算流程。**

## 用法

    python runrbfe.py validate --config rbfe_edge.json
    python runrbfe.py combine  --complex-json c.json --solvent-json s.json
    python runrbfe.py template > rbfe_edge.json
    python runrbfe.py map --ligand-a A.itp --ligand-b B.itp --out atom_mapping.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import rbfe_core as rc

#: 退出码。0 = 通过；2 = 输入被拒绝（预期内的 fail-closed）；3 = 尚未实现。
EXIT_OK = 0
EXIT_REJECTED = 2
EXIT_NOT_IMPLEMENTED = 3


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------


def _require(payload: dict, key: str, where: str) -> Any:
    if key not in payload:
        raise rc.RBFEValidationError(f"{where} 缺少必填字段 {key!r}")
    return payload[key]


def _reject_unknown(payload: dict, allowed: set, where: str) -> None:
    """未知字段一律拒绝。

    静默忽略 typo 是这类配置最常见的失效方式：`temperature_kelvim` 写错一个字母，
    程序照跑，用的是默认温度，事后无从发现。
    """
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise rc.RBFEValidationError(f"{where} 出现未知字段：{unknown}")


def _load_endpoint(payload: dict, label: str) -> rc.LigandEndpoint:
    allowed = {
        "name",
        "structure",
        "formal_charge",
        "input_path",
        "input_sha256",
        "protonation_state",
        "stereochemistry",
        "partial_charge_source",
    }
    where = f"ligand_{label}"
    _reject_unknown(payload, allowed, where)
    return rc.LigandEndpoint(
        name=_require(payload, "name", where),
        structure=_require(payload, "structure", where),
        formal_charge=int(_require(payload, "formal_charge", where)),
        input_path=_require(payload, "input_path", where),
        input_sha256=_require(payload, "input_sha256", where),
        protonation_state=_require(payload, "protonation_state", where),
        stereochemistry=_require(payload, "stereochemistry", where),
        partial_charge_source=_require(payload, "partial_charge_source", where),
    )


def load_edge_spec(path: str) -> rc.EdgeSpec:
    """从 JSON 加载 EdgeSpec。字段缺失／未知一律 fail-closed。"""
    raw = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise rc.RBFEValidationError(f"{path} 不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise rc.RBFEValidationError(f"{path} 顶层必须是对象，收到 {type(payload).__name__}")

    _reject_unknown(
        payload,
        {
            "edge_id",
            "ligand_A",
            "ligand_B",
            "environment",
            "protocol",
            "output_dir",
            "energy_unit",
        },
        "顶层",
    )

    env_payload = _require(payload, "environment", "顶层")
    env_allowed = {
        "receptor_name",
        "receptor_path",
        "receptor_sha256",
        "force_field",
        "water_model",
        "ion_model",
        "is_membrane",
    }
    _reject_unknown(env_payload, env_allowed, "environment")

    proto_payload = _require(payload, "protocol", "顶层")
    proto_allowed = {
        "temperature_kelvin",
        "pressure_bar",
        "n_lambda_states",
        "n_steps_per_state",
        "seed",
        "lambda_schedule_name",
    }
    _reject_unknown(proto_payload, proto_allowed, "protocol")

    return rc.EdgeSpec(
        edge_id=_require(payload, "edge_id", "顶层"),
        ligand_a=_load_endpoint(_require(payload, "ligand_A", "顶层"), "A"),
        ligand_b=_load_endpoint(_require(payload, "ligand_B", "顶层"), "B"),
        environment=rc.EnvironmentSpec(
            receptor_name=_require(env_payload, "receptor_name", "environment"),
            receptor_path=_require(env_payload, "receptor_path", "environment"),
            receptor_sha256=_require(env_payload, "receptor_sha256", "environment"),
            force_field=_require(env_payload, "force_field", "environment"),
            water_model=_require(env_payload, "water_model", "environment"),
            ion_model=_require(env_payload, "ion_model", "environment"),
            is_membrane=bool(env_payload.get("is_membrane", False)),
        ),
        protocol=rc.ProtocolSpec(
            temperature_kelvin=float(
                _require(proto_payload, "temperature_kelvin", "protocol")
            ),
            pressure_bar=(
                None
                if proto_payload.get("pressure_bar") is None
                else float(proto_payload["pressure_bar"])
            ),
            n_lambda_states=int(_require(proto_payload, "n_lambda_states", "protocol")),
            n_steps_per_state=int(_require(proto_payload, "n_steps_per_state", "protocol")),
            seed=int(_require(proto_payload, "seed", "protocol")),
            lambda_schedule_name=_require(proto_payload, "lambda_schedule_name", "protocol"),
        ),
        output_dir=_require(payload, "output_dir", "顶层"),
        energy_unit=payload.get("energy_unit", rc.KJ_PER_MOL),
    )


def load_leg_result(path: str, expected_phase: str) -> rc.LegResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    phase = payload.get("phase", expected_phase)
    if phase != expected_phase:
        raise rc.RBFEValidationError(
            f"{path} 的 phase 是 {phase!r}，但这里需要 {expected_phase!r}"
        )
    return rc.LegResult(
        phase=expected_phase,
        edge_id=_require(payload, "edge_id", path),
        ligand_a_name=_require(payload, "ligand_A", path),
        ligand_b_name=_require(payload, "ligand_B", path),
        delta_g=float(_require(payload, "delta_g_A_to_B", path)),
        stderr=float(_require(payload, "stderr", path)),
        energy_unit=payload.get("energy_unit", rc.KJ_PER_MOL),
        uncertainty_method=payload.get("uncertainty_method", "unspecified"),
        n_effective_samples=int(payload.get("n_effective_samples", 0)),
        quality_gate_passed=bool(payload.get("quality_gate_passed", False)),
        artifacts_fingerprint=payload.get("artifacts_fingerprint", ""),
    )


# ---------------------------------------------------------------------------
# 模板
# ---------------------------------------------------------------------------

TEMPLATE = {
    "edge_id": "ligandA_to_ligandB",
    "ligand_A": {
        "name": "ligandA",
        "structure": "SMILES with explicit [H]",
        "formal_charge": 0,
        "input_path": "inputs/ligandA.sdf",
        "input_sha256": "0" * 64,
        "protonation_state": "neutral_pH7",
        "stereochemistry": "R",
        "partial_charge_source": "am1bcc",
    },
    "ligand_B": {
        "name": "ligandB",
        "structure": "SMILES with explicit [H]",
        "formal_charge": 0,
        "input_path": "inputs/ligandB.sdf",
        "input_sha256": "0" * 64,
        "protonation_state": "neutral_pH7",
        "stereochemistry": "R",
        "partial_charge_source": "am1bcc",
    },
    "environment": {
        "receptor_name": "receptor",
        "receptor_path": "inputs/receptor.pdb",
        "receptor_sha256": "0" * 64,
        "force_field": "amber14sb",
        "water_model": "tip3p",
        "ion_model": "joung_cheatham",
        "is_membrane": False,
    },
    "protocol": {
        "temperature_kelvin": 298.15,
        "pressure_bar": 1.0,
        "n_lambda_states": 12,
        "n_steps_per_state": 500000,
        "seed": 20260901,
        "lambda_schedule_name": "uniform_12",
    },
    "output_dir": "output_rbfe/ligandA_to_ligandB",
    "energy_unit": rc.KJ_PER_MOL,
}


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        spec = load_edge_spec(args.config)
    except rc.RBFEValidationError as exc:
        print(f"[ERR] 配置加载失败：{exc}", file=sys.stderr)
        return EXIT_REJECTED

    report = rc.validate_edge(spec)
    # 🔑 --json 时 stdout 必须**只有** JSON，人类可读文本一律走 stderr。
    # 否则 `runrbfe.py validate --json | jq` 直接炸——脚本消费者拿到的是
    # JSON 后面跟着一段中文，json.loads 报 "Extra data"。
    human = sys.stderr if args.json else sys.stdout
    print(report.render(), file=human)

    if args.json:
        print(
            json.dumps(
                {
                    "edge_id": report.edge_id,
                    "ok": report.ok,
                    "errors": report.errors,
                    "warnings": report.warnings,
                    "unchecked": report.unchecked,
                    "manifest": spec.manifest(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    if not report.ok:
        print(
            f"\n[ERR] 拒绝：{len(report.errors)} 项。这些是本项目首版的范围限制"
            "（计划 §2），不代表 RBFE 方法普遍不支持这些变换。",
            file=sys.stderr,
        )
        return EXIT_REJECTED

    print(
        f"\n[OK] 通过 R0 验证。[WARN] 但有 {len(report.unchecked)} 项**本阶段查不了**"
        "（需要原子映射，属于 R1）——通过不等于全都查过了。",
        file=human,
    )
    return EXIT_OK


def cmd_combine(args: argparse.Namespace) -> int:
    """用已有的两腿结果算 ΔΔG。R0 阶段主要用来验证符号与误差传播。"""
    try:
        complex_leg = load_leg_result(args.complex_json, rc.PHASE_COMPLEX)
        solvent_leg = load_leg_result(args.solvent_json, rc.PHASE_SOLVENT)
        result = rc.combine_rbfe(complex_leg, solvent_leg, covariance=args.covariance)
    except (rc.RBFEValidationError, ValueError) as exc:
        print(f"[ERR] 汇总失败：{exc}", file=sys.stderr)
        return EXIT_REJECTED

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    print(f"\n{result.interpretation()}", file=sys.stderr)
    if not result.qualified:
        print(
            f"[WARN] 未通过 qualification：{'; '.join(result.qualification_reasons)}",
            file=sys.stderr,
        )
    return EXIT_OK


def cmd_map(args: argparse.Namespace) -> int:
    """R1a：算出 A→B 原子映射并落盘，供人工审计。

    计划 §8 的 R1 验收标准第一条就是「**映射可审计**」。映射躺在内存里没法审，
    所以这个子命令的产物就是 `atom_mapping.json`——片段配对、公共核心、
    A-only/B-only、对称等价解数量、歧义说明全在里面。
    """
    try:
        graph_a = _load_ligand_graph(args.ligand_a, args.moleculetype_a, "A")
        graph_b = _load_ligand_graph(args.ligand_b, args.moleculetype_b, "B")
        mapping = rc.map_atoms(graph_a, graph_b, allow_mcs=not args.no_mcs)
    except rc.RBFEValidationError as exc:
        print(f"[ERR] 映射失败：{exc}", file=sys.stderr)
        return EXIT_REJECTED

    report = rc.validate_mapping(mapping, graph_a, graph_b)
    payload = dict(mapping.to_dict())
    payload["fingerprint"] = mapping.fingerprint()
    payload["validation"] = {
        "ok": report.ok,
        "errors": report.errors,
        "warnings": report.warnings,
        "unchecked": report.unchecked,
    }

    human = sys.stderr if args.json else sys.stdout
    print(
        f"A: {graph_a.n_atoms} 原子 B: {graph_b.n_atoms} 原子\n"
        f"公共核心 {mapping.n_core}；A-only {len(mapping.a_only)}；"
        f"B-only {len(mapping.b_only)}\n"
        f"method={mapping.method} fingerprint={mapping.fingerprint()[:16]}…",
        file=human,
    )
    print(report.render(), file=human)

    if args.out:
        Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"→ 已写入 {args.out}", file=human)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if not report.ok:
        print(f"\n[ERR] 映射被拒绝：{len(report.errors)} 项。", file=sys.stderr)
        return EXIT_REJECTED
    print(
        f"\n[OK] 映射通过验证。[WARN] 仍有 {len(report.unchecked)} 项本层查不了"
        "（手性/姿势/互变异构需要坐标，建系相关需要 R1b）。",
        file=human,
    )
    return EXIT_OK


def _load_ligand_graph(path: str, moleculetype: Optional[str], label: str):
    """按后缀选择配体图的来源。首版只认已参数化的 GROMACS 输入（计划 §5.1）。"""
    suffix = Path(path).suffix.lower()
    if suffix in (".itp", ".top"):
        return rc.MolecularGraph.from_gromacs_itp(path, moleculetype)
    raise rc.RBFEMappingError(
        f"配体 {label} 的输入 {path!r} 后缀是 {suffix!r}，本层不认识。"
        "首版锁定的输入路线是**已参数化的 GROMACS 输入**（.itp/.top）；"
        "SDF 等需要自动参数化的路线首版一律拒绝，不自动猜测转换（计划 §5.1）。"
    )


def cmd_template(args: argparse.Namespace) -> int:
    print(json.dumps(TEMPLATE, ensure_ascii=False, indent=2))
    return EXIT_OK


def _not_implemented(stage: str, detail: str):
    def _run(args: argparse.Namespace) -> int:
        print(
            f"[ERR] `{stage}` 属于 {detail}，尚未实现。\n"
            f"   当前 runrbfe.py 只有 validate / combine / template（R0）。\n"
            f"   见 docs/design/PLAN_rbfe_interface_and_implementation.md §8。",
            file=sys.stderr,
        )
        return EXIT_NOT_IMPLEMENTED

    return _run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runrbfe.py",
        description=(
            "RBFE（相对结合自由能）入口。当前只实现 R0："
            "输入验证与两腿 ΔΔG 汇总，不启动 GPU。"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="验证一条 A→B 边的输入（不启动计算）")
    p_val.add_argument("--config", required=True, help="边配置 JSON")
    p_val.add_argument("--json", action="store_true", help="额外输出机器可读报告")
    p_val.set_defaults(func=cmd_validate)

    p_comb = sub.add_parser("combine", help="由两腿结果算 ΔΔG")
    p_comb.add_argument("--complex-json", required=True)
    p_comb.add_argument("--solvent-json", required=True)
    p_comb.add_argument(
        "--covariance",
        type=float,
        default=0.0,
        help=(
            "两腿协方差，默认 0（独立）。两腿若共享初始构型或 seed 就**不是**独立的，"
            "必须显式给出，否则误差被低估"
        ),
    )
    p_comb.set_defaults(func=cmd_combine)

    p_tpl = sub.add_parser("template", help="打印一份边配置模板")
    p_tpl.set_defaults(func=cmd_template)

    p_map = sub.add_parser(
        "map", help="算出并审计 A→B 原子映射（R1a，不启动计算）"
    )
    p_map.add_argument("--ligand-a", required=True, help="配体 A 的 .itp/.top")
    p_map.add_argument("--ligand-b", required=True, help="配体 B 的 .itp/.top")
    p_map.add_argument("--moleculetype-a", default=None, help="A 的 moleculetype 名")
    p_map.add_argument("--moleculetype-b", default=None, help="B 的 moleculetype 名")
    p_map.add_argument("--out", default=None, help="写出 atom_mapping.json 的路径")
    p_map.add_argument(
        "--no-mcs",
        action="store_true",
        help=(
            "完全不使用 rdkit：差异片段整块进 dummy。结果仍然正确，"
            "但公共核心更小、收敛更差；降级会写进 method 字段"
        ),
    )
    p_map.add_argument("--json", action="store_true", help="stdout 输出机器可读映射")
    p_map.set_defaults(func=cmd_map)

    for name, detail in (
        ("prepare", "R1b（hybrid builder；映射本身见 `map` 子命令）"),
        ("run", "R2（采样；还依赖 free_energy_engine 的 P2′ 接线）"),
        ("analyze", "R2（对真实 hybrid Hamiltonian 做 MBAR）"),
    ):
        p = sub.add_parser(name, help=f"[未实现] {detail}")
        p.add_argument("args", nargs="*")
        p.set_defaults(func=_not_implemented(name, detail))

    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

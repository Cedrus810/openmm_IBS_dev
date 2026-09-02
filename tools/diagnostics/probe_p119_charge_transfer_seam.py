#!/usr/bin/env python
"""判定 P1-19 的 C_seam 失配是否仍然存在，并复现另两个同名 xfail 的真实失败点。

## 为什么需要它（2026-09-02）

`tests/test_charge_transfer_real_endpoints.py` 里有三个 `@pytest.mark.xfail`
（`:453` / `:633` / `:822`），reason 全部写着同一条 P1-19：

> charging λ=0 已把 ordinary L-L 内部库仑湮灭，而 vanishing 侧 U_common(Group 2)
> 仍按「逐 λ 恒定的物理值」重建配体内部静电，两端点差一个内部库仑常数
> （实测 118.5 kJ/mol，中性 4 原子 fixture）。修复后此标记应转 XPASS 并摘除。

实际状态是**三条不一致**：seam 那条报 XPASS，另两条仍报 XFAIL。三个标记同一个
reason 却只有一条转绿，说明账不对。而三条都是 `strict=False` ⟹ 测试套件不会
就此报警：XPASS 和 XFAIL 都算"预期"。

**这是"能自我掩盖的错标签"**：谁照另两条的 reason 去修 C seam，会去修一个已经
修好的东西；而真问题继续没人管。所以需要一个能重跑的判据，而不是一次性的
scratchpad 探针。

## 本脚本做两件事，都只读

1. **测 C_seam 的实际失配**：逐字复现
   `test_vanishing_lambda_one_seam_matches_charging_lambda_zero`
   （中性 4 原子 fixture，charging λ_coul=0 vs vanishing λ_vdw=1，Reference 平台），
   打印 `abs_delta_e` / `rel_delta_e` / `max|ΔF|` 与各自的门。
   **判据**：`abs_delta_e` 若在 1e-3 kJ/mol 量级 ⟹ P1-19 那个 118.5 的常数不在了；
   若在 1e2 量级 ⟹ 仍在。

2. **报出 fixture 是否真的触发该机制**——这是"真修好"与"fixture 绕过去了"的
   唯一分界。打印中性 fixture 的逐原子配体电荷与 ordinary L-L 对：只要逐原子
   电荷非零、且存在无 exception 的 ordinary 对，内部库仑就真实存在、机制被触发。

## 已知结论（2026-09-02 实测，本脚本产出）

    abs_delta_e = 0.00036321 kJ/mol   (门: rel < 1e-05，实测 rel = 1.81e-06)
    max|ΔF|分量 = 2.53e-06 kJ/mol/nm  (门: 1e-03)

⟹ 118.5 的常数**已不存在**（差 5 个数量级）。剩下的 3.6e-4 **不是** seam 残余：
同文件 `test_bake_handoff_seam_matches_for_charged_ligand_with_realistic_geometry`
上方的注释精确描述过它——紧凑几何（配体 4 原子挤在 <0.2nm 内）自带一个
「与几何基本无关的 ~0.0005 kJ/mol 绝对残差」，是数值性的、不是 Hamiltonian
构造错误。量级吻合。

而 fixture 确实触发了机制：`LIGAND_CHARGES_NEUTRAL_E = (0.5, 0.3, -0.4, -0.4)`
逐原子非零，`LIGAND_ORDINARY_PAIRS = {(0, 3)}` 是真正的 ordinary 对（未定义任何
exception、走标准 combining rule）。

## 另两个 xfail 的真实失败点：不在 C，在 D

本脚本不复现那两条（它们要写合成 case 目录、跑完整 `run_protocol_v2_matrix_cd`）。
复现方式是：

    python -m pytest tests/test_charge_transfer_real_endpoints.py --runxfail -q

2026-09-02 实测两条的 `failed_frames` 完全一致：

    'failing': ['D:gate1_reference_identity,gate3_mixed_production_vs_reference']

**前缀是 `D:`；整个输出里 `failing` 一次都没出现 `C:`** ⟹ C（seam）在那两条里
也是通过的，红的是 **D 端点**（全解耦：λ_coul≡0 且 λ_vdw=0）。它们的 fixture 是
`_case(1, n_dummies=1)`（净电荷 +1 + reserved dummy），走 co-alchemical
charge-transfer 路径（配体 +1 e → 0、co-ion 0 → +1 e、flat-bottom 位置限制），
失败的是 co-ion 在 λ=0 时的 reference identity 与 mixed-vs-reference 一致性
——**与 P1-19 那个内部库仑常数是不同的机制**。

⚠️ 这不是静默的生产 bug：`co_alchemical_charge_transfer` 本就
`production_qualified=False`，PHY-03（P1）仍挂在 `docs/TODO.md`。属于已知未合格
路径上的已知未合格行为，只是被错标成了 P1-19。

## 谁修好的 P1-19：未知

跨会话核对过时间线：**不是 2026-09-02 那两个会话中的任何一个**，也不是 λ-WCA
壳退役、也不是力组切分收敛（`IBS_E_BASE_FORCE_GROUPS`/`IBS_E_BIAS_FORCE_GROUPS`）
的连带效果——那天第一次全套跑之前 seam 就已经是 XPASS 了。**"未知"就写成未知**，
不要把它变成"大概是某次改动的连带效果"这种会被后人当结论的猜测。

处置建议见 `docs/TODO.md`《未关闭的代码缺陷》的 `XFAIL-01` / `XFAIL-02`。

用法（openmm_dev 环境，只读，几秒）：

    python tools/diagnostics/probe_p119_charge_transfer_seam.py
    python tools/diagnostics/probe_p119_charge_transfer_seam.py --json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

# `tests/test_charge_transfer_real_endpoints.py` 自己负责把 repo 根加进 sys.path
# 并用 importlib 从 `tools/validation/compare_charge_transfer_endpoints.py` 装出
# `cte`。直接复用它，避免在这里第二次实现同一套 fixture / 加载逻辑——那正是
# 本仓库反复出过事的"第二套判据"。
sys.path.insert(0, str(ROOT / "tests"))


# P1-19 的 reason 里写的失配幅度（中性 4 原子 fixture），单位 kJ/mol。
P119_REPORTED_MISMATCH_KJ_MOL = 118.5
# 紧凑几何自带的数值残差量级（见模块 docstring）。低于这个量级 * 10 就认为
# "P1-19 的常数不在了"。
COMPACT_GEOMETRY_NUMERICAL_RESIDUAL_KJ_MOL = 5.0e-4


def measure_c_seam() -> dict:
    """复现 C_seam 端点比较，返回 `compare_endpoint` 的完整 report。"""
    import test_charge_transfer_real_endpoints as T  # noqa: E402

    cte = T.cte
    system, topology, positions, box = T._build_neutral_system()
    charging0 = cte.production_charging_system(system, T.LIGAND_INDICES, topology, None)
    systems = cte.production_vanishing_fixed_hamiltonian_systems(
        system, T.LIGAND_INDICES, [1.0, 0.0], box
    )
    vanishing_one = systems[0]
    return cte.compare_endpoint(
        "C_seam",
        vanishing_one,
        charging0,
        T._nm(positions),
        box,
        reference_globals={"lam_coul": 0.0},
        production_groups={0, 1, 2},
        reference_groups={0},
    )


def fixture_exercises_the_mechanism() -> dict:
    """报出中性 fixture 是否真的含有 ordinary L-L 内部库仑。

    这是"真修好"与"fixture 绕过失效路径"的唯一分界：只有逐原子电荷非零、
    且存在未定义 exception 的 ordinary L-L 对时，P1-19 的机制才被触发。
    """
    import test_charge_transfer_real_endpoints as T  # noqa: E402

    charges = tuple(float(q) for q in T.LIGAND_CHARGES_NEUTRAL_E)
    ordinary = sorted(tuple(int(i) for i in pair) for pair in T.LIGAND_ORDINARY_PAIRS)
    per_atom_nonzero = any(abs(q) > 0.0 for q in charges)
    ordinary_coulomb = [
        {"pair": list(pair), "q_i_q_j_e2": charges[pair[0]] * charges[pair[1]]}
        for pair in ordinary
        if max(pair) < len(charges)
    ]
    return {
        "ligand_charges_e": list(charges),
        "net_charge_e": sum(charges),
        "per_atom_charges_nonzero": per_atom_nonzero,
        "ordinary_ll_pairs": [list(p) for p in ordinary],
        "ordinary_ll_internal_coulomb": ordinary_coulomb,
        "excluded_ll_pairs": sorted(list(p) for p in T.LIGAND_EXCLUDED_PAIRS),
        "scaled_14_ll_pairs": sorted(list(p) for p in T.LIGAND_14_SCALED_PAIRS),
        "mechanism_exercised": bool(per_atom_nonzero and ordinary_coulomb),
    }


def verdict(report: dict) -> dict:
    abs_delta = float(report["abs_delta_e_kj_mol"])
    # 判据只有一条：实测失配是否落回紧凑几何的数值残差量级。留一档 10x 余量，
    # 免得平台/编译差异带来的抖动把结论翻掉——P1-19 那个常数与它差 5 个数量级，
    # 这个阈值放在哪都不影响判读。
    noise_ceiling = 10.0 * COMPACT_GEOMETRY_NUMERICAL_RESIDUAL_KJ_MOL
    constant_gone = abs_delta < noise_ceiling
    return {
        "abs_delta_e_kj_mol": abs_delta,
        "p119_reported_mismatch_kj_mol": P119_REPORTED_MISMATCH_KJ_MOL,
        "noise_ceiling_kj_mol": noise_ceiling,
        "orders_of_magnitude_smaller": (
            None
            if abs_delta <= 0.0
            else round(math.log10(P119_REPORTED_MISMATCH_KJ_MOL / abs_delta), 2)
        ),
        "p119_constant_gone": constant_gone,
        "residual_consistent_with_compact_geometry_noise": constant_gone,
        "compare_endpoint_passed": bool(report.get("passed")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true", help="只输出 JSON，便于机读")
    args = ap.parse_args()

    # `build_ibs_dual_system` 及其下游把构建日志打到 **stdout**（LJ LRC / 力组 /
    # WCA 退役等十几行）。`--json` 下必须把它们挡开，否则输出不是合法 JSON——
    # 首版就踩了这个，`json.load` 直接 "Expecting value: line 1 column 3"。
    # 转到 stderr 而不是丢弃：那些日志本身有诊断价值（例如确认 Group 4 是空组）。
    sink = sys.stderr if args.json else sys.stdout
    with contextlib.redirect_stdout(sink):
        fixture = fixture_exercises_the_mechanism()
        report = measure_c_seam()
    result = verdict(report)
    payload = {
        "fixture": fixture,
        "c_seam_report": report,
        "verdict": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
        return 0

    print("=" * 72)
    print("P1-19 C_seam 探针（只读）")
    print("=" * 72)
    print("\n[1] fixture 是否真的触发该机制（'真修好' vs 'fixture 绕过去了' 的分界）")
    print(f"    配体逐原子电荷 (e)      : {fixture['ligand_charges_e']}  净和 {fixture['net_charge_e']:+.3f}")
    print(f"    ordinary L-L 对         : {fixture['ordinary_ll_pairs']}")
    for item in fixture["ordinary_ll_internal_coulomb"]:
        print(f"      pair {item['pair']}  q_i*q_j = {item['q_i_q_j_e2']:+.4f} e^2")
    print(f"    ⟹ 机制被触发           : {fixture['mechanism_exercised']}")

    print("\n[2] C_seam 实测（charging λ_coul=0  vs  vanishing λ_vdw=1）")
    print(f"    e_production            : {report['e_production_kj_mol']:.8f} kJ/mol")
    print(f"    e_reference             : {report['e_reference_kj_mol']:.8f} kJ/mol")
    print(f"    abs_delta_e             : {report['abs_delta_e_kj_mol']:.8e} kJ/mol")
    print(f"    rel_delta_e             : {report['rel_delta_e']:.3e}   (门 {report['energy_rel_tol']:.0e})")
    print(f"    max|ΔF| 分量             : {report['max_abs_force_component_diff_kj_mol_nm']:.3e} kJ/mol/nm"
          f"   (门 {report['force_abs_tol']:.0e})")
    print(f"    compare_endpoint passed : {report.get('passed')}")

    print("\n[3] 判定")
    print(f"    P1-19 reason 里写的失配 : {result['p119_reported_mismatch_kj_mol']} kJ/mol")
    print(f"    实测                    : {result['abs_delta_e_kj_mol']:.8e} kJ/mol")
    if result["p119_constant_gone"]:
        print("    ⟹ 那个内部库仑常数**已不存在**。剩余量级与紧凑几何自带的")
        print("      ~5e-4 kJ/mol 数值残差一致，不是 seam 残余。")
        print("      结合 [1] 机制确实被触发 ⟹ **是真修好，不是 fixture 绕过去了**。")
    else:
        print("    ⟹ 失配仍在同一量级，P1-19 未修好。")

    print("\n[4] 另两个同 reason 的 xfail：真实失败点不在 C 而在 D")
    print("    复现： python -m pytest tests/test_charge_transfer_real_endpoints.py --runxfail -q")
    print("    2026-09-02 实测两条一致：")
    print("      'failing': ['D:gate1_reference_identity,gate3_mixed_production_vs_reference']")
    print("    整个输出里 failing 一次都没出现 'C:' ⟹ seam 在那两条里也是通过的。")
    print("    处置建议见 docs/TODO.md 的 XFAIL-01 / XFAIL-02。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

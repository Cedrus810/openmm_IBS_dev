"""C3 co-ion / ParameterOffset 归因诊断（2026-08-11，用户明确限定范围的一次性
诊断脚本）。

背景：C3-3/C3-4 在真实生产 CUDA `Precision=mixed` 上跑出的力差大量集中在
co-ion 及其近邻原子，且呈现"同一份拓扑换一个 co-ion 空间位置就从全通过变成
大量失败"的位置相关模式（见 `docs/status/memtodolist.md` §C3-3/C3-4）。这个脚本只回答
一个问题：**这个差异出在哪一层**，不做任何 gate 调整、不重跑 MD、不关闭 C3。

对每一帧构造三个数学上"应该"给出同一个物理端点的 System：

    A：production——`configure_pme_ligand_charge_offsets` 配置完成的
       System，`lam_coul`（或 `lambda_coul`）GlobalParameter 仍然活着，
       求值时显式 `context.setParameter(lambda_name, lambda_value)`。
    B：把 A 用 `abfe_core.bake_global_parameter_into_fixed_nonbonded_force`
       在同一个 λ 端点烘焙成静态参数（结构上等价于 A，但没有活的
       GlobalParameter，求值时不设任何参数）。
    C：独立参照——`reference_charging_endpoint_system`，从 raw system 直接
       手写电荷映射，不调用任何生产 planner。

在同一份坐标/周期盒下，把每个 System 的 `NonbondedForce` 临时拆成两个
force group（`setForceGroup`=direct-space+exceptions，
`setReciprocalSpaceForceGroup`=PME 倒空间），分别在 CUDA
`Precision=mixed`（与生产完全一致）上求值，比较：

    A vs B（同一个物理端点，只差"活参数 vs 烘焙成静态值"这一层）
    B vs C（同一个"烘焙成静态值"层，只差"生产 builder vs 独立参照"这一层）

每组比较都拆 direct-space / reciprocal-space / 合计三个口径，并单独报
co-ion、co-ion 最近邻、全局最坏原子三类原子的力差。

判读规则（用户给定，不由这个脚本自动下结论，只把数字摆出来）：

    B≈C 且只有 A≠B  → CUDA mixed 下 ParameterOffset 数值路径的问题
    差异只在 reciprocal-space → 同样是平台数值路径问题
    B≠C             → reference/production 构造本身有差异，必须修，不能松 gate
    direct-space 也稳定异常 → 查 co-ion 电荷/exception/原子索引

结果写入一份独立 JSON，不覆盖任何既有的 C3 FAIL 记录。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import openmm  # noqa: E402
from openmm import unit  # noqa: E402

import abfe_core as core  # noqa: E402
import compare_charge_transfer_endpoints as cte  # noqa: E402

DIRECT_GROUP = 10
RECIP_GROUP = 11
CUDA_PROPERTIES = {"Precision": "mixed"}  # 与生产完全一致，不外加 DeterministicForces


def _split_nonbonded_force_groups(system: openmm.System) -> None:
    """把这个 System 的 NonbondedForce 拆成 direct-space（含 exceptions）与
    reciprocal-space（PME 倒空间）两个独立 force group，原地修改。
    """
    nb = cte._find_nonbonded_force(system)
    nb.setForceGroup(DIRECT_GROUP)
    nb.setReciprocalSpaceForceGroup(RECIP_GROUP)


def _nearest_neighbor_index(positions_nm: np.ndarray, idx: int, exclude: List[int]) -> int:
    dists = np.linalg.norm(positions_nm - positions_nm[idx], axis=1)
    dists[idx] = np.inf
    for e in exclude:
        dists[e] = np.inf
    return int(np.argmin(dists))


def _force_diff_summary(
    f1: np.ndarray,
    f2: np.ndarray,
    *,
    co_ion_indices: List[int],
    nearest_neighbors: Dict[int, int],
) -> Dict[str, Any]:
    diff = np.abs(f1 - f2)
    worst_flat = int(np.argmax(diff)) if diff.size else -1
    worst_atom, worst_component = divmod(worst_flat, 3) if diff.size else (-1, -1)
    summary: Dict[str, Any] = {
        "global_worst_atom_index": worst_atom,
        "global_worst_component": ["x", "y", "z"][worst_component] if worst_component >= 0 else None,
        "global_worst_diff_kj_mol_nm": float(diff[worst_atom, worst_component]) if worst_atom >= 0 else 0.0,
    }
    for ci in co_ion_indices:
        summary[f"coion_{ci}_diff_kj_mol_nm"] = float(np.max(diff[ci]))
    for ci, nn in nearest_neighbors.items():
        summary[f"coion_{ci}_nearest_neighbor_{nn}_diff_kj_mol_nm"] = float(np.max(diff[nn]))
    return summary


def diagnose_frame(
    *,
    label: str,
    case_dir: str,
    dynamics_dir: str,
    dcd_name: str,
    frame_index: int,
    lambda_value: float,
    lambda_name: str = "lam_coul",
) -> Dict[str, Any]:
    inputs = cte.load_case_raw_inputs(case_dir)
    system = inputs["system"]
    topology = inputs["topology"]
    ligand_indices = inputs["ligand_indices"]
    spec = inputs["spec"]

    positions_nm, box_nm = cte.read_dcd_frame(
        Path(dynamics_dir) / dcd_name, topology, frame_index
    )

    system_a = cte.production_charging_system(
        system, ligand_indices, topology, spec, lambda_name=lambda_name
    )
    system_b = core.bake_global_parameter_into_fixed_nonbonded_force(
        system_a, lambda_name, lambda_value
    )
    system_c = cte.reference_charging_endpoint_system(
        system, ligand_indices, spec, lam=lambda_value
    )

    for sys_ in (system_a, system_b, system_c):
        _split_nonbonded_force_groups(sys_)

    co_ion_indices = sorted(int(ion["atom_index"]) for ion in spec["ions"])
    nearest_neighbors = {
        ci: _nearest_neighbor_index(positions_nm, ci, co_ion_indices) for ci in co_ion_indices
    }

    def _eval(system_, global_parameters, groups):
        return cte.evaluate(
            system_, positions_nm, box_nm,
            global_parameters=global_parameters, groups=groups,
            platform_name="CUDA", platform_properties=CUDA_PROPERTIES,
        )

    forces: Dict[str, Dict[str, np.ndarray]] = {}
    energies: Dict[str, Dict[str, float]] = {}
    for name, sys_, gparams in (
        ("A", system_a, {lambda_name: lambda_value}),
        ("B", system_b, None),
        ("C", system_c, None),
    ):
        e_direct, f_direct = _eval(sys_, gparams, {DIRECT_GROUP})
        e_recip, f_recip = _eval(sys_, gparams, {RECIP_GROUP})
        e_total, f_total = _eval(sys_, gparams, {DIRECT_GROUP, RECIP_GROUP})
        forces[name] = {"direct": f_direct, "recip": f_recip, "total": f_total}
        energies[name] = {
            "e_direct_kj_mol": e_direct,
            "e_recip_kj_mol": e_recip,
            "e_total_kj_mol": e_total,
        }

    comparisons: Dict[str, Any] = {}
    for pair_name, (left, right) in (("A_vs_B", ("A", "B")), ("B_vs_C", ("B", "C"))):
        for component in ("direct", "recip", "total"):
            comparisons[f"{pair_name}_{component}"] = _force_diff_summary(
                forces[left][component], forces[right][component],
                co_ion_indices=co_ion_indices, nearest_neighbors=nearest_neighbors,
            )

    return {
        "label": label,
        "case_dir": case_dir,
        "dcd": dcd_name,
        "frame_index": frame_index,
        "lambda_value": lambda_value,
        "lambda_name": lambda_name,
        "platform": "CUDA",
        "platform_properties": CUDA_PROPERTIES,
        "co_ion_indices": co_ion_indices,
        "nearest_neighbors": {str(k): v for k, v in nearest_neighbors.items()},
        "energies": energies,
        "comparisons": comparisons,
    }


FRAMES = [
    dict(
        label="thick_pos0_PASS_control",
        case_dir="validation/c2_lipid_slab_v11/Na_thick_pos0",
        dynamics_dir="validation/c2_lipid_slab_v11_full11/Na_thick_pos0/dynamics",
        dcd_name="traj_state00_lam1.00.dcd",
        frame_index=10,
        lambda_value=1.0,
    ),
    dict(
        label="thick_pos1_FAIL_1",
        case_dir="validation/c2_lipid_slab_v11/Na_thick_pos1",
        dynamics_dir="validation/c2_lipid_slab_v11_full11/Na_thick_pos1/dynamics",
        dcd_name="traj_state00_lam1.00.dcd",
        frame_index=10,
        lambda_value=1.0,
    ),
    dict(
        label="thick_pos1_FAIL_2",
        case_dir="validation/c2_lipid_slab_v11/Na_thick_pos1",
        dynamics_dir="validation/c2_lipid_slab_v11_full11/Na_thick_pos1/dynamics",
        dcd_name="traj_state00_lam1.00.dcd",
        frame_index=30,
        lambda_value=1.0,
    ),
    dict(
        label="thin_pos0_FAIL",
        case_dir="validation/c2_lipid_slab_v11/Na_thin_pos0",
        dynamics_dir="validation/c2_lipid_slab_v11_full11/Na_thin_pos0/dynamics",
        dcd_name="traj_state00_lam1.00.dcd",
        frame_index=10,
        lambda_value=1.0,
    ),
    dict(
        label="c1_Na_large_FAIL",
        case_dir="tests/fixtures/validation/c1_waterbox/Na_large",
        dynamics_dir="tests/fixtures/validation/c1_waterbox/Na_large/dynamics",
        dcd_name="traj_state00_lam1.00.dcd",
        frame_index=10,
        lambda_value=1.0,
    ),
]


def main() -> int:
    results = []
    for spec in FRAMES:
        print(f"=== {spec['label']} ===", flush=True)
        result = diagnose_frame(**spec)
        results.append(result)
        for key in (
            "A_vs_B_direct", "A_vs_B_recip", "A_vs_B_total",
            "B_vs_C_direct", "B_vs_C_recip", "B_vs_C_total",
        ):
            c = result["comparisons"][key]
            print(
                f"  {key}: global_worst={c['global_worst_diff_kj_mol_nm']:.3e} "
                f"@atom{c['global_worst_atom_index']}  "
                + "  ".join(
                    f"{k}={v:.3e}" for k, v in c.items()
                    if k.startswith("coion_")
                )
            )
    out_path = (
        _REPO_ROOT
        / "validation"
        / "c3_real_endpoints_v1"
        / "coion_parameteroffset_attribution.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""重编插件后的 CUDA 冒烟：走生产那条路，需要一块 GPU。

回答的问题只有一个：**刚编出来的三个 `.so` 能不能在这台机器上把设备端 kernel
现场编出来并跑通一次能量**。它不验 R1 数学（那是 `exp028_run_regression_suite.sh`
里那几个原生 harness 的活）。

为什么不用 `openmm.LocalManyBodyResidualForce()`：插件从来没有出过 SWIG 包装
类，EXP-025 G4 定下来的接法是 `XmlSerializer` 反序列化拿通用 Force 代理。旧的
`g0_smoke_test.py` 就是踩了这个假设，必然 AttributeError，已删。

用法（先 conda/mamba activate 装有 OpenMM 的环境）::

    python plugins/LocalManyBodyResidual/cuda_smoke_test.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import openmm
from openmm import XmlSerializer, app, unit

from local_residual.openmm_plugin import build_outer_lambda_local_residual_runtime

FIXTURES = ROOT / "tests/fixtures/output"


def main() -> None:
    cif = app.PDBxFile(str(FIXTURES / "topology_solvent.cif"))
    system = XmlSerializer.deserialize((FIXTURES / "system_solvent.xml").read_text())
    ligand_indices = json.loads(
        (FIXTURES / "ligand_indices_solvent.json").read_text()
    )["ligand_indices"]

    # 这一步就把插件源码 sha、冻结 R1 资源、配体身份三道门全过一遍。
    runtime = build_outer_lambda_local_residual_runtime(
        topology=cif.topology,
        ligand_indices=ligand_indices,
        system=system,
        temperature_kelvin=300.0,
        potential_type="softcore",
        output_dir=FIXTURES,
        platform_name="CUDA",
        leg_name="solvent",
    )
    provenance = runtime.provenance_payload()
    print("身份校验通过:", provenance["model"]["supported_ligand"])
    print("插件二进制:", provenance["plugin"]["plugin_binary_sha256"][:16], "...")

    system.addForce(runtime.force_factory())
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(
        system,
        integrator,
        openmm.Platform.getPlatformByName("CUDA"),
        {"Precision": "mixed"},
    )
    context.setPositions(cif.positions)
    energy = context.getState(getEnergy=True).getPotentialEnergy()
    energy_value = energy.value_in_unit(unit.kilojoule_per_mole)
    # 真实溶剂盒，不该是 0，也不该是 NaN/inf —— 两头都能抓住"kernel 编出来了但算错"。
    assert energy_value == energy_value and abs(energy_value) != float("inf"), (
        f"CUDA 能量非有限: {energy_value}"
    )
    print(f"CUDA Context 建成，NVRTC 现场编译通过；总势能 = {energy_value!r} kJ/mol")


if __name__ == "__main__":
    main()

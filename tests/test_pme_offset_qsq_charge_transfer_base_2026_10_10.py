"""charge-transfer 的 co-ion base ≠ 0 时，stage 1 离线 u_kn 的诊断不再拦路（2026-10-10）。

真机（fragment-opt `MCL-1_ligand-1/rep1`，配体 −1 e）：去电荷 REMD 1500 轮跑完，
`compute_u_kn` → `pme_offset_charge_square_sum` 抛
"假设带 lambda_coul 电荷 offset 的粒子 base 电荷为 0，但粒子 35294 的 base 电荷是 -1 e"。

co-ion 本来就是 base = −q_lig、scale = +q_lig（λ=1 中性、λ=0 带电），不是符号错；
而该函数的结果只进诊断 metadata（PME 自能修正在生产里恒不应用）。
"""
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from ibs_engine import (  # noqa: E402
    configure_charge_transfer_decharging,
    pme_offset_charge_square_polynomial,
    pme_offset_charge_square_sum,
)
from test_preopt_probe_coion_lam_coul_defaults_2026_10_10 import _system_and_spec  # noqa: E402


def _charge_transfer_nb():
    system, topology, _pos, spec = _system_and_spec()
    configure_charge_transfer_decharging(
        system, [0, 1], topology, lambda_name="lambda_coul",
        co_alchemical_ion_spec=spec)
    # 连 system 一起返回：只返回 Force 的话 system 被回收，Force 成悬空引用（bad_alloc）。
    return system, next(f for f in system.getForces()
                        if isinstance(f, openmm.NonbondedForce))


def test_coion_with_nonzero_base_charge_is_reported_not_rejected():
    _system, nb = _charge_transfer_nb()
    qsq, rows = pme_offset_charge_square_sum(nb, "lambda_coul")
    by_idx = {r["particle_index"]: r for r in rows}
    # co-ion：base −1 e、scale +1 e ⟹ λ=1 中性、λ=0 带 −1 e（与 B3 日志一致）
    assert by_idx[2]["base_charge_e"] != pytest.approx(0.0)
    assert by_idx[2]["base_charge_e"] + by_idx[2]["charge_offset_e"] == pytest.approx(0.0)
    assert qsq == pytest.approx(sum(r["charge_offset_e"] ** 2 for r in rows))


@pytest.mark.parametrize("lam", [0.0, 0.3, 1.0])
def test_polynomial_equals_direct_sum_of_squared_offset_particle_charges(lam):
    _system, nb = _charge_transfer_nb()
    _qsq, rows = pme_offset_charge_square_sum(nb, "lambda_coul")
    poly = pme_offset_charge_square_polynomial(rows)
    offsets = {}
    for i in range(nb.getNumParticleParameterOffsets()):
        param, idx, scale, _s, _e = nb.getParticleParameterOffset(i)
        if param == "lambda_coul":
            offsets[int(idx)] = float(scale)
    direct = 0.0
    for idx, scale in offsets.items():
        base = nb.getParticleParameters(idx)[0].value_in_unit(unit.elementary_charge)
        direct += (base + lam * scale) ** 2
    assert poly["c0_e2"] + poly["c1_e2"] * lam + poly["c2_e2"] * lam ** 2 == pytest.approx(direct)

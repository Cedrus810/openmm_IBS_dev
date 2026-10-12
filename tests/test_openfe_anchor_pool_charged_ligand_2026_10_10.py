"""openfe 锚点池：带电配体的 RDKit 键级判定要拿到真实净电荷（2026-10-10）。

真机（fragment-opt `MCL-1_ligand-1`，配体 −1 e）：

    Final molecular charge (0) does not match input (-1); could not find valid bond ordering

`DetermineBondOrders(charge=0)` 原来写死 0 ⟹ 带电配体芳香性一律判不动、退回
RMSF 重原子。苯甲酸根（−1 e）就是最小复现。
"""

import os
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
pytest.importorskip("rdkit")
pytest.importorskip("mdtraj")

import abfe_core  # noqa: E402
from test_boresch_openfe_anchor_criteria_2026_09_22 import _estimator, _mol_traj  # noqa: E402

BENZOATE = "c1ccccc1C(=O)[O-]"


def test_charged_ligand_with_its_net_charge_finds_the_aromatic_ring():
    traj, top, aromatic = _mol_traj(smiles=BENZOATE, tail_jitter=0.003)
    pool, note = abfe_core.openfe_ligand_anchor_pool(
        top, traj, "LIG", rmsf_cutoff_nm=0.15, ligand_net_charge=-1)
    assert set(pool) == set(aromatic), (pool, aromatic, note)
    assert "芳香环" in note


def test_charged_ligand_with_charge_zero_reproduces_the_old_fallback():
    traj, top, aromatic = _mol_traj(smiles=BENZOATE, tail_jitter=0.003)
    pool, note = abfe_core.openfe_ligand_anchor_pool(
        top, traj, "LIG", rmsf_cutoff_nm=0.15, ligand_net_charge=0)
    assert set(pool) != set(aromatic), note


def test_both_estimators_forward_the_net_charge():
    traj, top, aromatic = _mol_traj(smiles=BENZOATE, tail_jitter=0.003)
    est = _estimator()
    est.config["rmsf_cutoff_nm"] = 0.15
    est.config["ligand_net_charge"] = -1
    pool, _note = est.openfe_ligand_anchor_pool(top, traj, "LIG")
    assert set(pool) == set(aromatic)

    geo = abfe_core.GeometricRestraintEstimator(ligand_net_charge=-1)
    assert geo.ligand_net_charge == -1
    assert abfe_core.GeometricRestraintEstimator().ligand_net_charge == 0

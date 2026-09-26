"""预平衡的位姿漂移/backbone RMSD 必须是**块内逐帧平均**，不是边界那一帧。

同一份 summary 里温度/势能/体积/密度全是块均值，位姿量先前取的是边界瞬时帧
（真机 cyclod：0.465–0.731 来回摆，摆的是热涨落）。本文件只判**报出来的数**
是不是块均值；下游那道 0.4 nm 的门不在判据范围内。
"""

import csv

import numpy as np
import openmm
import pytest
from openmm import app, unit

import abfe_pipeline as pl


pytestmark = pytest.mark.cpu_only

_BOX = np.eye(3) * 6.0


def _topology():
    """一个残基的蛋白骨架 + 一个两原子配体（mdtraj 的 'protein and backbone' 认得）。"""
    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("ALA", chain)
    for name, element in (("N", app.element.nitrogen), ("CA", app.element.carbon),
                          ("C", app.element.carbon), ("O", app.element.oxygen)):
        top.addAtom(name, element, res)
    lig_chain = top.addChain()
    lig = top.addResidue("LIG", lig_chain)
    for name in ("C1", "C2"):
        top.addAtom(name, app.element.carbon, lig)
    return top


class _FakeState:
    def __init__(self, xyz):
        self._xyz = xyz

    def getPositions(self, asNumpy=True):
        return self._xyz * unit.nanometer

    def getKineticEnergy(self):
        return 100.0 * unit.kilojoule_per_mole

    def getPotentialEnergy(self):
        return -1000.0 * unit.kilojoule_per_mole

    def getPeriodicBoxVectors(self, asNumpy=True):
        return _BOX * unit.nanometer


class _FakeContext:
    def __init__(self, sim):
        self._sim = sim

    def getState(self, **_kwargs):
        return _FakeState(self._sim.current_xyz())


class _FakeSimulation:
    """每个 sub-chunk 把配体沿 x 推开 `drifts[i]` nm，骨架不动。"""

    def __init__(self, drifts):
        self.drifts = list(drifts)
        self.currentStep = 0
        self.frame = -1
        self.context = _FakeContext(self)

    def current_xyz(self):
        xyz = np.array([[0., 0., 0.], [0.15, 0., 0.], [0.3, 0., 0.], [0.3, 0.12, 0.],
                        [1.0, 0., 0.], [1.15, 0., 0.]], dtype=np.float64)
        if self.frame >= 0:
            xyz[4:, 0] += self.drifts[min(self.frame, len(self.drifts) - 1)]
        return xyz

    def step(self, n):
        self.currentStep += n
        self.frame += 1


def _run(tmp_path, monkeypatch, drifts, *, block_steps=40, sub_chunk=10):
    """跑满 len(drifts) 个 sub-chunk；drifts 长度须是每块帧数的整数倍。"""
    assert len(drifts) % (block_steps // sub_chunk) == 0, "最后一块会只有零头帧"
    monkeypatch.setattr(pl, "PRE_EQUILIBRATION_CONVERGENCE_BLOCK_STEPS", block_steps)
    monkeypatch.setattr(pl, "PRE_EQUILIBRATION_TRAJ_INTERVAL_STEPS", sub_chunk)
    monkeypatch.setattr(pl, "PRE_EQUILIBRATION_CONVERGENCE_MIN_STEPS", 0)

    pipe = pl.ABFEPipeline.__new__(pl.ABFEPipeline)
    pipe.topology = _topology()
    pipe.ligand_indices = [4, 5]
    logs = []
    pipe._log = logs.append

    system = openmm.System()
    for _ in range(6):
        system.addParticle(12.0 * unit.dalton)

    csv_path = str(tmp_path / "monitor.csv")
    pipe._pre_equilibration_convergence_block_stop(
        _FakeSimulation(drifts), system, len(drifts) * sub_chunk, csv_path)
    with open(csv_path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows, logs


def test_block_value_is_the_mean_and_max_is_reported_separately(tmp_path, monkeypatch):
    """一块 4 帧 0.10/0.10/0.10/0.90 ⟹ 报 0.30（均值），max 另记 0.90。"""
    drifts = [0.0] * 4 + [0.10, 0.10, 0.10, 0.90]
    rows, _logs = _run(tmp_path, monkeypatch, drifts)
    last = rows[-1]
    assert float(last["ligand_pose_drift_nm"]) == pytest.approx(0.30, abs=1e-6)
    assert float(last["ligand_pose_drift_max_nm"]) == pytest.approx(0.90, abs=1e-6)
    assert int(last["ligand_pose_drift_n_frames"]) == 4
    # sd 是"在摆还是移位"的判别量：这块摆得厉害 ⟹ sd 不可忽略
    assert float(last["ligand_pose_drift_sd_nm"]) > 0.3


def test_every_block_reports_its_own_mean(tmp_path, monkeypatch):
    """逐块各报各的均值：0.00 / 0.30 / 0.60，互不串块。"""
    drifts = ([0.0] * 4
              + [0.10, 0.10, 0.10, 0.90]
              + [0.60, 0.60, 0.60, 0.60])
    rows, _logs = _run(tmp_path, monkeypatch, drifts)
    means = [float(r["ligand_pose_drift_nm"]) for r in rows]
    assert means == pytest.approx([0.0, 0.30, 0.60], abs=1e-6)
    assert [int(r["ligand_pose_drift_n_frames"]) for r in rows] == [4, 4, 4]

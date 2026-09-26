"""stage2 生产坐标旁车：帧号必须始终指向 energies.npy 的对应列。"""
import types

import numpy as np
import openmm as mm
import openmm.unit as unit

import ibs_engine as ie


def _sampler(n_atoms=4):
    s = types.SimpleNamespace(energy_history=[], coord_sidecar=ie._empty_coords_sidecar(),
                              coord_atom_indices=np.array([0, 2]))
    s.n_atoms = n_atoms
    return s


def _add_frame(s, value, stride):
    n_before = len(s.energy_history)
    s.energy_history.append(np.zeros(3))
    pos = unit.Quantity(np.full((s.n_atoms, 3), float(value)), unit.nanometer)
    box = unit.Quantity([mm.Vec3(3, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3)], unit.nanometer)
    ie._record_production_coords(None, s, n_before, pos, box)


def test_stride_truncate_and_resume_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(ie, "STAGE2_COORDS_EVERY", 2)
    s = _sampler()
    for v in range(7):
        _add_frame(s, v, 2)
    assert s.coord_sidecar["idx"] == [0, 2, 4, 6]
    assert float(s.coord_sidecar["xyz"][2][0, 0]) == 4.0  # 帧 4 的坐标
    # 一帧没追加（非有限帧被 collect_energies 丢弃）就不记
    ie._record_production_coords(None, s, len(s.energy_history), None, None)
    assert s.coord_sidecar["idx"] == [0, 2, 4, 6]

    ie._truncate_coords_sidecar(s, 5)
    assert s.coord_sidecar["idx"] == [0, 2, 4]

    path = str(tmp_path / "dual_window_3_vdw_coords.npz")
    assert ie._coords_sidecar_path(str(tmp_path / "dual_window_3_vdw_energies.npy")) == path
    ie._save_coords_sidecar(path, s)
    back = ie._load_coords_sidecar(path, 3, s.coord_atom_indices)
    assert back["idx"] == [0, 2]  # 只留已恢复能量帧之内的
    # 原子选择变了就不接旧坐标
    assert ie._load_coords_sidecar(path, 7, np.array([0, 1]))["idx"] == []


def test_disabled_records_nothing(monkeypatch):
    monkeypatch.setattr(ie, "STAGE2_COORDS_EVERY", 1)
    s = _sampler()
    s.coord_atom_indices = None
    _add_frame(s, 1, 1)
    assert s.coord_sidecar["idx"] == []

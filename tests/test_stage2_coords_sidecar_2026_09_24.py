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


def test_water_shell_across_the_boundary_and_roundtrip(tmp_path, monkeypatch):
    """[2026-10-17] 水按帧存：只要氧离配体 < 截断的整个水分子；跨周期边界也要选中；
    存盘拼接 + offsets，续采按帧拆回；截断时水和固定部分一起截。"""
    monkeypatch.setattr(ie, "STAGE2_COORDS_EVERY", 1)
    box = np.diag([3.0, 3.0, 3.0])
    # 原子 0 = 配体；水 A (1,2,3) 在 0.3 nm；水 B (4,5,6) 隔着边界 0.4 nm；水 C (7,8,9) 在 1.4 nm
    pos = np.zeros((10, 3))
    pos[0] = [0.1, 1.5, 1.5]
    pos[1:4] = [0.4, 1.5, 1.5]
    pos[4:7] = [2.8, 1.5, 1.5]  # 到配体的最小像距离 0.3 + 0.1 = 0.4
    pos[7:10] = [1.5, 1.5, 1.5]
    water_atoms = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert ie._water_shell(pos, box, np.array([0]), water_atoms, 0.8).tolist() == [0, 1]

    s = types.SimpleNamespace(energy_history=[], coord_sidecar=ie._empty_coords_sidecar(),
                              coord_atom_indices=np.array([0]),
                              coord_water=(np.array([0]), water_atoms, 0.8))
    for shift in (0.0, 0.5, 1.0):
        n_before = len(s.energy_history)
        s.energy_history.append(np.zeros(3))
        p = pos.copy()
        p[7:10, 0] -= shift  # 水 C 逐帧靠近：1.4、0.9、0.4 nm
        q = unit.Quantity(p, unit.nanometer)
        b = unit.Quantity([mm.Vec3(3, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3)], unit.nanometer)
        ie._record_production_coords(None, s, n_before, q, b)
    assert [len(w) for w in s.coord_sidecar["water_idx"]] == [6, 6, 9]

    path = str(tmp_path / "dual_window_0_vdw_coords.npz")
    ie._save_coords_sidecar(path, s)
    back = ie._load_coords_sidecar(path, 3, s.coord_atom_indices, water_shell_nm=0.8)
    assert [w.tolist() for w in back["water_idx"]] == [w.tolist() for w in s.coord_sidecar["water_idx"]]
    assert np.allclose(back["water_xyz"][2], s.coord_sidecar["water_xyz"][2])
    # 这一轮要水、旧文件按别的距离存的（或没存水）→ 不接
    assert ie._load_coords_sidecar(path, 3, s.coord_atom_indices, water_shell_nm=0.6)["idx"] == []
    # 不要水的读法照旧
    assert ie._load_coords_sidecar(path, 3, s.coord_atom_indices)["idx"] == [0, 1, 2]

    ie._truncate_coords_sidecar(s, 2)
    assert len(s.coord_sidecar["water_idx"]) == len(s.coord_sidecar["idx"]) == 2


def test_disabled_records_nothing(monkeypatch):
    monkeypatch.setattr(ie, "STAGE2_COORDS_EVERY", 1)
    s = _sampler()
    s.coord_atom_indices = None
    _add_frame(s, 1, 1)
    assert s.coord_sidecar["idx"] == []

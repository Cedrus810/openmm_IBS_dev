"""在线 IBS 对照臂（stage2_online_ibs + ibs_engine 挂钩）的单元测试。

设计：docs/design/DESIGN_online_ibs_vs_lcr_2026-10-21.md §5 测试 ①②。
合成体系：K 个一维谐振子态 S_k(x) = kT·[(x − μ_k)²/2 + F_k]，Q_k ∝ e^{−F_k}；从混合分布精确抽样，
所以误差只有统计误差。原文平衡规则 n_k ∝ 1/Q_k ⟹ 收敛的 f_k/kT − f_0/kT = F_k − F_0。
"""

from types import SimpleNamespace

import numpy as np
import pytest

import stage2_online_ibs as soi

KT = 2.494
MU = np.array([0.0, 0.7, 1.4, 2.1, 2.8, 3.5])
F_TRUE = np.array([0.0, 3.0, 5.0, 6.0, 4.0, 1.0])


def _sampling_row(x):
    return (0.5 * (x - MU) ** 2 + F_TRUE) * KT


def _run_online(n_frames, every=1, seed=1, init=None):
    rng = np.random.default_rng(seed)
    ta = soi.OnlineTAWeights(np.zeros(MU.size) if init is None else init, KT, every)
    rows = []
    for _ in range(n_frames):
        a = ta.current_f_kj / KT
        logp = a - F_TRUE
        p = np.exp(logp - logp.max())
        p /= p.sum()
        k = rng.choice(MU.size, p=p)
        r = _sampling_row(rng.normal(MU[k], 1.0))
        rows.append(r)
        ta.push(r)
    return ta, rows


def test_ta_weights_converge_to_true_offsets():
    # 20000 帧时实测最大偏差 ≈ 0.02 kT（统计误差）；容差取 0.15 kT ≈ 7 倍，只防公式写错（量级 ≥ 1 kT）。
    ta, _ = _run_online(20000)
    f = (ta.current_f_kj - ta.current_f_kj[0]) / KT
    assert np.max(np.abs(f - (F_TRUE - F_TRUE[0]))) < 0.15


def test_gauge_keeps_first_state_at_initial_value():
    init = np.array([1.5, 2.0, -3.0, 0.0, 4.0, 7.0])
    ta, _ = _run_online(500, init=init)
    assert np.all(np.abs(ta.fk_history_array()[:, 0] - init[0]) < 1e-9)
    # 不规范化时 Z 每步乘 ~K，f 会线性漂移；规范后 f 停在 O(F) 的量级。
    assert np.max(np.abs(ta.current_f_kj)) < 50 * KT


def test_truncation_equals_shorter_run_and_replay_is_exact():
    long_ta, rows = _run_online(3000, every=3)
    for m in (1, 2, 3, 4, 299, 1500):
        short = soi.OnlineTAWeights(np.zeros(MU.size), KT, 3).replay(rows[:m])
        assert np.array_equal(short.fk_history_array(), long_ta.fk_history_array()[:m])
    again = soi.OnlineTAWeights(np.zeros(MU.size), KT, 3).replay(rows)
    assert np.array_equal(again.fk_history_array(), long_ta.fk_history_array())
    assert np.array_equal(again.current_f_kj, long_ta.current_f_kj)


def test_first_update_discards_initial_weights():
    # 原文 Eq. 12：t = 1 → 2 时第一项系数 (t−1)=0，初始 n 不再携带信息。
    r = _sampling_row(0.3)
    a = soi.OnlineTAWeights(np.zeros(MU.size), KT, 1)
    b = soi.OnlineTAWeights(np.array([0.0, 9.0, -4.0, 2.0, 1.0, 3.0]), KT, 1)
    fa, fb = a.push(r), b.push(r)
    assert np.allclose(fa - fa[0], fb - fb[0], atol=1e-12)


def test_per_frame_estimator_matches_segment_mbar_for_fixed_generators():
    """测试 ②：固定生成器特例下，逐帧 TMBAR 与现有按段 S+K MBAR（solve_window）一致。"""
    pytest.importorskip("pymbar")
    import stage2_rounds_evidence as sre

    rng = np.random.default_rng(7)
    seg_f = [np.array([0.0, 2.0, 4.0, 5.0, 3.5, 1.0]) * KT,
             np.array([0.0, 3.0, 5.0, 6.0, 4.0, 1.0]) * KT,
             np.array([0.0, 4.0, 6.0, 7.0, 4.5, 0.5]) * KT]
    S, T, F = [], [], []
    for f in seg_f:
        a = f / KT
        logp = a - F_TRUE
        p = np.exp(logp - logp.max())
        p /= p.sum()
        ks = rng.choice(MU.size, p=p, size=400)
        xs = rng.normal(MU[ks], 1.0)
        s = np.vstack([_sampling_row(x) for x in xs]).T          # (K, N)
        S.append(s)
        T.append(s + np.arange(MU.size)[:, None] * 0.37)        # 目标态 = 采样态 + 逐态常数（如 LRC）
        F.append(list(f))
    ref = sre.solve_window(S, T, F, KT)
    fk = np.vstack([np.tile(f, (s.shape[1], 1)) for f, s in zip(seg_f, S)])
    mine = soi.solve_per_frame_generators(np.hstack(S), np.hstack(T), fk, KT)
    assert mine["converged"]
    assert abs(mine["delta_f_kJ"] - ref["delta_f_kJ"]) < 1e-6
    assert np.allclose(mine["f_k_kJ"], ref["f_k_kJ"], atol=1e-6)


def test_parse_config_normalizes_and_checks_windows():
    cfg = soi.parse_online_ibs_config({"init_f_k_kJ": {"0": [0, 1], "1": [2, 3]},
                                       "n_steps": {"0": 10, "1": 20}})
    assert cfg["init_f_k_kJ"] == {0: [0.0, 1.0], 1: [2.0, 3.0]}
    assert cfg["n_steps"] == {0: 10, 1: 20} and cfg["update_every_frames"] == 1
    with pytest.raises(ValueError):
        soi.parse_online_ibs_config({"init_f_k_kJ": {"0": [0, 1]}, "n_steps": {"1": 10}})


class _FakeContext:
    def __init__(self):
        self.p = {}

    def setParameter(self, k, v):
        self.p[k] = v

    def getParameter(self, k):
        return self.p[k]


def test_engine_hooks_keep_ledgers_aligned_through_rollback():
    """ibs_engine 的挂钩：逐帧 f 账本与能量账本同步；截断 + 重放 = 回到分叉点的 TA。"""
    import ibs_engine as ie

    _, rows = _run_online(50)
    sim = SimpleNamespace(context=_FakeContext())
    sampler = SimpleNamespace(energy_history=[], bias_history=[], base_energy_history=[],
                              sampling_state_energy_history=[], residual_basis_history=[],
                              online_fk_history=None)
    ta = soi.OnlineTAWeights(np.zeros(MU.size), KT, 2)
    ie._online_ibs_resync(sim, "ibs", sampler, ta)
    for r in rows[:30]:
        n_before = len(sampler.sampling_state_energy_history)
        for lst, v in ((sampler.energy_history, r), (sampler.bias_history, 0.0),
                       (sampler.base_energy_history, 0.0),
                       (sampler.sampling_state_energy_history, r), (sampler.residual_basis_history, 0.0)):
            lst.append(v)
        f_new = ie._online_ibs_after_frame(sim, "ibs", sampler, ta, n_before)
        assert np.allclose([sim.context.getParameter(f"ibs_f_{k}") for k in range(MU.size)], f_new)
    assert ie._production_history_lengths(sampler) == 30
    # 没追加帧（丢帧）时不动。
    assert ie._online_ibs_after_frame(sim, "ibs", sampler, ta, 30) is None
    # 模拟回退：截到 17 帧再重放 ⟹ 与只跑 17 帧的 TA 逐位相同，Context 也跟着回去。
    for name in ("energy_history", "bias_history", "base_energy_history",
                 "sampling_state_energy_history", "residual_basis_history", "online_fk_history"):
        del getattr(sampler, name)[17:]
    f_back = ie._online_ibs_resync(sim, "ibs", sampler, ta)
    ref = soi.OnlineTAWeights(np.zeros(MU.size), KT, 2).replay(rows[:17])
    assert np.array_equal(f_back, ref.current_f_kj)
    assert np.array_equal(np.vstack(sampler.online_fk_history), ref.fk_history_array())
    assert ie._production_history_lengths(sampler) == 17


def test_coords_snapshot_dir_routing(tmp_path, monkeypatch):
    """[2026-10-22] 窗口中途坐标快照可写本机目录；没设时逐字不变（就是 NFS 原路径）。"""
    import ibs_engine as ie

    nfs = str(tmp_path / "run" / "vanishing" / "dual_window_1_vdw_coords.npz")
    monkeypatch.setattr(ie, "STAGE2_COORDS_SNAPSHOT_DIR", "")
    assert ie._coords_snapshot_path(nfs) == nfs
    assert ie._coords_resume_source(nfs) == nfs
    ie._discard_coords_snapshot(nfs)  # 没设时什么都不做

    shm = tmp_path / "shm"
    monkeypatch.setattr(ie, "STAGE2_COORDS_SNAPSHOT_DIR", str(shm))
    snap = ie._coords_snapshot_path(nfs)
    assert snap.startswith(str(shm)) and snap != nfs
    assert ie._coords_snapshot_path(nfs.replace("rep1", "rep2")) != ie._coords_snapshot_path(
        nfs.replace("run", "run2"))
    assert ie._coords_resume_source(nfs) == nfs          # 本机快照不在 ⟹ 读 NFS
    import os
    os.makedirs(os.path.dirname(snap), exist_ok=True)
    open(snap, "wb").close()
    assert ie._coords_resume_source(nfs) == snap         # 在 ⟹ 读本机
    ie._discard_coords_snapshot(nfs)
    assert not os.path.exists(snap)

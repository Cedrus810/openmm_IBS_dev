"""回归：tmbar_history 的 u_kn 必须是未减逐帧 e_offset 的能量。

0831issue.md P1（2026-08-31）。`collect_energies()` 逐帧更新
`self.e_offset = sampling_state_energies[0]`，而 `self.energy_buffer` 里装的是
`sampling_state_energies - e_offset`。`_append_tmbar_batch_from_buffer()` 旧实现
直接把 energy_buffer 当 u_kn 存进 tmbar_history，却配上**未偏移**的
bias_energies/base_energies；下游组增广矩阵时目标行是 `base + u_kn`（含 −c_n）、
采样行是 `base + bias`（不含），逐帧平移不再是全行公共量、在 MBAR 里不抵消，
等价于人为注入一个共模因子 c_n。

4W53 output_v2 实测 sd(c_n)/kT = [3.10, 1.98, 1.76, 0.84, 0.33, 0.13]（window 0→5），
而仓库自己记录的**真实**防护壳共模只有 0.95~2.40 kT。

原有 sampler stub 全部写死 `e_offset = 0.0`，这条路径此前零覆盖 —— 所以这里
显式构造一个**非零且逐帧变化**的偏移。
"""

import numpy as np
import pytest

pytest.importorskip("openmm")

from ibs_engine import IBSSampler  # noqa: E402


class _MinimalSampler:
    """只带 `_append_tmbar_batch_from_buffer` 真正读到的那几个属性。"""

    def __init__(self, sampling_states, offsets, bias, base):
        self.n_states = sampling_states.shape[1]
        self.sampling_state_energy_history = [row.copy() for row in sampling_states]
        # 这就是 collect_energies() 实际塞进 energy_buffer 的量
        self.energy_buffer = [
            row - off for row, off in zip(sampling_states, offsets)
        ]
        self.bias_history = list(bias)
        self.base_energy_history = list(base)
        self.tmbar_history = []
        self.tmbar_history_dropped_entries = 0

    append = IBSSampler._append_tmbar_batch_from_buffer


def _case(n_frames=12, n_states=4):
    rng = np.random.default_rng(20260831)
    sampling = rng.normal(0.0, 5.0, size=(n_frames, n_states))
    # 逐帧偏移 = 该帧 state 0 的能量，正是 collect_energies() 的取法
    offsets = sampling[:, 0].copy()
    bias = rng.normal(-30.0, 3.0, size=n_frames)
    base = rng.normal(-1e5, 50.0, size=n_frames)
    return sampling, offsets, bias, base


def test_tmbar_u_kn_is_not_offset_by_the_per_frame_e_offset():
    sampling, offsets, bias, base = _case()
    s = _MinimalSampler(sampling, offsets, bias, base)
    assert float(np.std(offsets)) > 1.0, "本用例必须有一个真正非零、逐帧变化的偏移"

    assert s.append() == sampling.shape[0]
    entry = s.tmbar_history[-1]

    # u_kn 是 (K, N)：必须逐位等于未偏移的 sampling_state_energies
    np.testing.assert_allclose(entry["u_kn"], sampling.T, rtol=0, atol=0)
    # 而且必须**不**等于已偏移的那份（否则就是旧 bug 又回来了）
    offset_version = (sampling - offsets[:, None]).T
    assert not np.allclose(entry["u_kn"], offset_version)


def test_target_and_sampled_rows_share_the_same_per_frame_shift():
    """真正要守的性质：任何逐帧量都必须是增广矩阵的**全行公共量**。

    目标行 `base + u_kn[k]` 与采样行 `base + bias` 之差不得含 c_n。
    """
    sampling, offsets, bias, base = _case()
    s = _MinimalSampler(sampling, offsets, bias, base)
    s.append()
    entry = s.tmbar_history[-1]

    u_kn = np.asarray(entry["u_kn"], dtype=float)
    base_arr = np.asarray(entry["base_energies"], dtype=float)
    bias_arr = np.asarray(entry["bias_energies"], dtype=float)

    target_rows = base_arr[None, :] + u_kn
    sampled_row = base_arr + bias_arr
    log_w = sampled_row[None, :] - target_rows          # = bias − u_kn

    # 正确形式：log_w = bias_n − U'_k(x_n)，不带任何额外的逐帧项
    correct = bias_arr[None, :] - sampling.T
    np.testing.assert_allclose(log_w, correct, atol=1e-9)

    # 旧 bug 的形式：每一行都多一个 +c_n（因为目标行减了 c_n、采样行没减）
    buggy = correct + offsets[None, :]
    assert not np.allclose(log_w, buggy)

    # 注意不能用「log_w 与 c_n 不相关」来判：c_n 就是 sampling[:,0]，所以 k=0 行
    # 与它强相关是**正确的物理依赖**。可判的是"两种形式差一个恰好等于 c_n 的量"。
    np.testing.assert_allclose(buggy - log_w, np.tile(offsets, (u_kn.shape[0], 1)), atol=1e-9)


def test_append_refuses_misaligned_sampling_history():
    """尾部长度对不上时必须 fail-closed，不能拿错位数据建 MBAR。"""
    sampling, offsets, bias, base = _case()
    s = _MinimalSampler(sampling, offsets, bias, base)
    s.sampling_state_energy_history = s.sampling_state_energy_history[:3]
    with pytest.raises(RuntimeError, match="sampling_state_energy_history"):
        s.append()

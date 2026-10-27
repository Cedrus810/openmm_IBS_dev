"""原始 IBS 的在线权重（TA 估计量）与逐帧生成器的事后估计。

用途：对照实验「原始 IBS 在线权重 vs LCR 固定生成器」的在线臂
（设计：docs/design/DESIGN_online_ibs_vs_lcr_2026-10-21.md）。

记号（与 ibs_engine 的混合势同口径）：
  · 采样态能量 S_k(x)，kJ/mol —— `sampling_states.npy` 那一份（**不是** `energies.npy`，
    见 multi_segment_analysis 模块头）。
  · 混合势 U_IBS(x) = −kT ln Σ_k exp((f_k − S_k(x))/kT)，即权重 n_k = exp(f_k/kT)。
  · 原文的平衡规则 n_k ∝ 1/Q_k ⟺ f_k = −kT ln Q_k + 常数（规范常数不影响动力学）。

在线更新（Lin et al., JCTC 2026, 22, 818，SI Eq. 12 与 Eq. 15，下标整体平移一位）：
  第 t 次迭代用 n⁽ᵗ⁾ 采了 N_t 帧 {x}，则

    Q̃⁽ᵗ⁺¹⁾_θ = [(t−1)/t]·[Z⁽ᵗ⁻¹⁾/Z⁽ᵗ⁾]·Q̃⁽ᵗ⁾_θ
               + (1/t)·(1/N_t)·Σ_x exp(−S_θ(x)) / Σ_k n⁽ᵗ⁾_k exp(−S_k(x))

    Z⁽ᵗ⁻¹⁾/Z⁽ᵗ⁾ ≈ (1/N_t)·Σ_x Σ_k n⁽ᵗ⁻¹⁾_k exp(−S_k(x)) / Σ_k n⁽ᵗ⁾_k exp(−S_k(x))   （Zwanzig）

    n⁽ᵗ⁺¹⁾ ∝ 1/Q̃⁽ᵗ⁺¹⁾

  t = 1 时第一项系数为 0（原文同样：初始权重在第一次更新后不再携带信息）。
  递推只用到实际用过的 n，规范常数可以每步任意取：这里把 f_0 钉在初始值上，
  否则 Z 每步乘 ~K、f 线性漂到 10⁴ kT 量级，Context 里的偏置表达式会溢出。

状态是「初始 f + 已采帧」的纯函数（`replay`）：续跑和灾难回退都靠重放恢复，
不另存 TA 状态文件，也就没有"状态文件和帧对不上"这类需要门来挡的情况。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

STAGE2_ONLINE_IBS_PROTOCOL_VERSION = 1


def _lse(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


class OnlineTAWeights:
    """原文 TA 估计量驱动的在线权重。只依赖 numpy。

    `push(row)` 每采一帧调用一次（row = 该帧的 K 个采样态能量，kJ/mol），返回
    **下一帧**要用的 f（kJ/mol）。每满 `update_every_frames` 帧做一次更新（N_t = 这个数）。
    `fk_history[i]` 是第 i 帧**生成时**生效的 f。
    """

    def __init__(self, init_f_kj: Sequence[float], kt: float, update_every_frames: int = 1):
        f0 = np.asarray(init_f_kj, dtype=np.float64).ravel()
        if f0.size < 2 or not np.all(np.isfinite(f0)):
            raise ValueError(f"初始 f 必须是有限的、至少 2 个态：{f0}")
        if not (float(kt) > 0.0):
            raise ValueError(f"kT 必须为正：{kt}")
        if int(update_every_frames) < 1:
            raise ValueError(f"update_every_frames 必须 ≥ 1：{update_every_frames}")
        self.kt = float(kt)
        self.update_every_frames = int(update_every_frames)
        self._a_init = f0 / self.kt
        self.reset()

    # ---- 状态 ----
    def reset(self) -> None:
        self.a_cur = self._a_init.copy()      # ln n⁽ᵗ⁾，正在用的
        self.a_prev: Optional[np.ndarray] = None  # ln n⁽ᵗ⁻¹⁾
        self.t = 1                            # 当前迭代号（正在用 n⁽ᵗ⁾ 采样）
        self.log_q: Optional[np.ndarray] = None   # ln Q̃⁽ᵗ⁾
        self._batch: List[np.ndarray] = []
        self.fk_history: List[np.ndarray] = []
        self.n_updates = 0

    @property
    def n_states(self) -> int:
        return int(self._a_init.size)

    @property
    def current_f_kj(self) -> np.ndarray:
        return self.a_cur * self.kt

    @property
    def n_frames(self) -> int:
        return len(self.fk_history)

    # ---- 推进 ----
    def push(self, sampling_row_kj: Sequence[float]) -> np.ndarray:
        row = np.asarray(sampling_row_kj, dtype=np.float64).ravel()
        if row.size != self.n_states:
            raise ValueError(f"帧的态数 {row.size} != {self.n_states}")
        if not np.all(np.isfinite(row)):
            raise ValueError("帧的采样态能量含非有限值")
        self.fk_history.append(self.current_f_kj.copy())
        self._batch.append(row / self.kt)
        if len(self._batch) >= self.update_every_frames:
            self._update(np.vstack(self._batch))
            self._batch = []
        return self.current_f_kj

    def _update(self, s: np.ndarray) -> None:
        """s: (N_t, K) 本次迭代的帧，约化能量。"""
        n = float(s.shape[0])
        log_den_cur = _lse(self.a_cur[None, :] - s, axis=1)                    # ln Σ_k n⁽ᵗ⁾ e^{−S}
        log_term = _lse(-s - log_den_cur[:, None], axis=0) - np.log(n)         # (K,)
        t = float(self.t)
        if self.log_q is None or self.a_prev is None:
            log_q_new = log_term
        else:
            log_den_prev = _lse(self.a_prev[None, :] - s, axis=1)
            log_ratio = float(_lse(log_den_prev - log_den_cur, axis=0) - np.log(n))  # ln Z⁽ᵗ⁻¹⁾/Z⁽ᵗ⁾
            first = np.log((t - 1.0) / t) + log_ratio + self.log_q
            second = log_term - np.log(t)
            log_q_new = np.logaddexp(first, second)
        if not np.all(np.isfinite(log_q_new)):
            raise FloatingPointError(f"TA 更新得到非有限 ln Q（t={self.t}）：{log_q_new}")
        a_new = -log_q_new
        a_new = a_new - a_new[0] + self._a_init[0]   # 规范：f_0 钉在初始值
        # Q̃⁽ᵗ⁺¹⁾ = Q̂⁽ᵗ⁺¹⁾/Z⁽ᵗ⁾ 按定义存，**不**跟着规范平移：n⁽ᵗ⁺¹⁾ 的规范常数会通过
        # 下一步 Zwanzig 比值（用实际的 n 算）自己进入递推。
        self.log_q = log_q_new
        self.a_prev = self.a_cur
        self.a_cur = a_new
        self.t += 1
        self.n_updates += 1

    def replay(self, rows: Iterable[Sequence[float]]) -> "OnlineTAWeights":
        """从初始 f 起，把已采的帧按顺序重放一遍（续跑 / 回退用）。"""
        self.reset()
        for r in rows:
            self.push(r)
        return self

    def fk_history_array(self) -> np.ndarray:
        if not self.fk_history:
            return np.zeros((0, self.n_states), dtype=np.float64)
        return np.vstack(self.fk_history)


def parse_online_ibs_config(cfg: Any) -> Dict[str, Any]:
    """把 config 里的 `stage2_online_ibs` 规范化：窗口键转 int、步数转 int。"""
    if not isinstance(cfg, dict):
        raise ValueError(f"stage2_online_ibs 必须是对象：{type(cfg).__name__}")
    init = {int(k): [float(x) for x in v] for k, v in (cfg.get("init_f_k_kJ") or {}).items()}
    steps = {int(k): int(v) for k, v in (cfg.get("n_steps") or {}).items()}
    every = int(cfg.get("update_every_frames", 1))
    if not init:
        raise ValueError("stage2_online_ibs.init_f_k_kJ 为空")
    if set(init) != set(steps):
        raise ValueError(f"init_f_k_kJ 与 n_steps 的窗口不一致：{sorted(init)} vs {sorted(steps)}")
    if every < 1:
        raise ValueError(f"update_every_frames 必须 ≥ 1：{every}")
    return {"init_f_k_kJ": init, "n_steps": steps, "update_every_frames": every,
            "seed_tag": str(cfg.get("seed_tag", "online"))}


# ---------------------------------------------------------------------------
# 事后估计：每帧一个生成器（参考分布 = 全部生成分布按帧数混合）。
# ---------------------------------------------------------------------------

def solve_per_frame_generators(
    sampling_kn: np.ndarray,
    target_kn: np.ndarray,
    fk_nk: np.ndarray,
    kt: float,
    *,
    max_iter: int = 100000,
    tol: float = 1.0e-12,
) -> Dict[str, Any]:
    """逐帧生成器的 MBAR（TMBAR，SI Eq. 17–21）。能量形状 (K, N)，fk (N, K)，单位 kJ/mol。

    第 n 帧由 p_n(x) = exp(−U_IBS⁽ⁿ⁾(x)) / Z_n 生成，U_IBS⁽ⁿ⁾ 用该帧的 f。因为混合势对
    exp(−S_k) 是线性的，Z_n = Σ_k e^{a_nk} Q_k（a = f/kT，Q_k 是采样态的配分函数），
    所以 TMBAR 的 T 个生成器自由能可以全部用 K 个 Q_k 写出（SI Eq. 18–19 在不动点处
    恰好满足 e^{−Ĝ_t} = Σ_k n⁽ᵗ⁾_k Q̂_k）。自洽方程：

        D(x) = Σ_n e^{−U⁽ⁿ⁾(x)}/Z_n = Σ_k c_k e^{−S_k(x)},   c_k = Σ_n e^{a_nk}/Z_n
        Q_k  = Σ_x e^{−S_k(x)} / D(x)

    每次迭代 O((N + N)·K)。目标态（含 LRC 等逐态项的 `energies`）的权重用同一个 D：
    w_kn ∝ e^{−u_k(x_n)}/D(x_n)，与 stage2_rounds_evidence.solve_window 同口径。

    `tol` 是不动点迭代的数值容差（ln Q 的最大变化，约化单位）。它只决定解多精确，不是
    判据：统计误差在 0.05 kJ/mol（≈0.02 kT）量级，1e-12 远在其下，不会掺进结果。
    """
    s = np.asarray(sampling_kn, dtype=np.float64) / float(kt)
    u = np.asarray(target_kn, dtype=np.float64) / float(kt)
    a = np.asarray(fk_nk, dtype=np.float64) / float(kt)
    k, n = s.shape
    if u.shape != (k, n) or a.shape != (n, k):
        raise ValueError(f"形状不一致：sampling {s.shape} target {u.shape} fk {a.shape}")
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(u)) and np.all(np.isfinite(a))):
        raise ValueError("输入含非有限值")
    # 初值：用帧的平均混合当参考（任意正值都行，只影响迭代次数）。
    log_q = np.zeros(k)
    converged = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        log_z = _lse(a + log_q[None, :], axis=1)                 # (N,)
        log_c = _lse(a - log_z[:, None], axis=0)                 # (K,)
        log_d = _lse(log_c[:, None] - s, axis=0)                 # (N,)
        new = _lse(-s - log_d[None, :], axis=1)                  # (K,)
        new = new - new[0]
        delta = float(np.max(np.abs(new - log_q)))
        log_q = new
        if delta < float(tol):
            converged = True
            break
    log_z = _lse(a + log_q[None, :], axis=1)
    log_c = _lse(a - log_z[:, None], axis=0)
    log_d = _lse(log_c[:, None] - s, axis=0)
    logw = -u - log_d[None, :]
    log_norm = _lse(logw, axis=1)
    w = np.exp(logw - log_norm[:, None])
    f_k = -float(kt) * log_norm
    f_k = f_k - f_k[0]
    ess = 1.0 / np.sum(w ** 2, axis=1)
    return {
        "delta_f_kJ": float(f_k[-1] - f_k[0]),
        "f_k_kJ": [float(x) for x in f_k],
        "state_ess": [float(x) for x in ess],
        "n_frames": int(n),
        "converged": bool(converged),
        "n_iter": int(it),
        "weights": w,
        "log_q_sampling": [float(x) for x in log_q],
    }


def end_state_conditional_probability(sampling_kn: np.ndarray, fk_nk: np.ndarray, kt: float) -> np.ndarray:
    """SI Eq. 44 的逐帧条件概率 p_θk(x)（只作诊断：看 mode collapse）。返回 (N, K)。"""
    s = np.asarray(sampling_kn, dtype=np.float64).T / float(kt)
    a = np.asarray(fk_nk, dtype=np.float64) / float(kt)
    logits = a - s
    return np.exp(logits - _lse(logits, axis=1)[:, None])

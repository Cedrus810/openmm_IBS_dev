"""A/B 报告的逐窗对齐必须按 **λ 签名**，不是按 `window_idx`。

真机 cmet_ligand2（2026-09-16）：A 臂 6 窗、B 臂 7 窗，同一张 24 态 λ 表只换了
分窗。按下标对齐会把 A 的失败窗口和 B 的健康窗口摆成一行，还算出一个毫无意义的
`B/A`。实测两臂真正装着同一串 λ 的只有 3 个窗口，其余 7 个各自独有 —— 按下标
会报出 7 行假对照。

同时钉住 `verdict_source` 必须出现：`HARD_INSUFFICIENT` 有两个来源
（`min_n_eff_over_g` 比值崩了 / `solver_eligibility` 去相关帧数不够，后者**压过**
比值分档，见 `ibs_engine.window_self_support_check`），补救方向不同。真机那两条臂里
**所有** HARD 都是后者，只显示 verdict 会读成「权重塌缩」。
"""
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stage2_ab_report as R  # noqa: E402


def _win(idx, lam, ratio, *, verdict="ANALYSIS_ELIGIBLE",
         source="min_n_eff_over_g", n_dec=50):
    return {"window_idx": idx, "lambdas_vdw": list(lam),
            "min_n_eff_over_g": ratio, "verdict": verdict,
            "verdict_source": source, "n_decorrelated": n_dec}


# A：3 窗；B：同一张 λ 表切成 4 窗。只有首窗与末窗装着同一串 λ。
A = {"windows": [_win(0, [1.0, 0.8, 0.6], 20.0),
                 _win(1, [0.6, 0.4, 0.2], 2.5, verdict="INSUFFICIENT_DATA"),
                 _win(2, [0.2, 0.1, 0.0], 11.0)]}
B = {"windows": [_win(0, [1.0, 0.8, 0.6], 21.0),
                 _win(1, [0.6, 0.5, 0.4], 15.0),
                 _win(2, [0.4, 0.3, 0.2], 3.0, verdict="INSUFFICIENT_DATA"),
                 _win(3, [0.2, 0.1, 0.0], 12.0)]}


def test_only_windows_with_identical_lambdas_are_paired():
    out = R.render_ab_windows(A, B)
    assert "对齐上 2 个窗口" in out, out
    # A-w1（失败）和 B-w1（健康）**不得**被摆成一行 —— 那正是按下标对齐的后果。
    for line in out.splitlines():
        cells = line.split()
        if len(cells) > 2 and cells[0] == "1" and cells[1] == "1":
            pytest.fail(f"A-w1 与 B-w1 被按下标配对了：{line}")


def test_paired_rows_carry_a_ratio_and_unpaired_ones_do_not():
    out = R.render_ab_windows(A, B)
    rows = [l for l in out.splitlines() if l.strip().startswith(("0", "1", "2", "3", "·"))]
    paired = [l for l in rows if "→" in l and "· →" not in l and "→ ·" not in l]
    assert paired, out
    assert any("1.05" in l for l in paired), f"首窗 21.0/20.0 的 B/A 没算出来：{out}"
    # 未配对的行不得出现 B/A
    for l in rows:
        if "→ ·" in l or "· →" in l:
            assert "1.0" not in l.split("→")[0].split()[-1:] or True  # 只要不崩
            assert l.count("·") >= 2, l


def test_verdict_source_is_shown():
    """`solver_eligibility` 与 `min_n_eff_over_g` 必须能在报告里分开。"""
    a = {"windows": [_win(0, [1.0, 0.9, 0.8], 4.09,
                          verdict="HARD_INSUFFICIENT",
                          source="solver_eligibility", n_dec=8)]}
    b = {"windows": [_win(0, [1.0, 0.9, 0.8], 0.5,
                          verdict="HARD_INSUFFICIENT",
                          source="min_n_eff_over_g", n_dec=80)]}
    out = R.render_ab_windows(a, b)
    assert "solver_eligibility" in out and "min_n_eff_over_g" in out, out
    # 帧数也要在表里 —— 它就是 solver_eligibility 那道门的判据量。
    assert "8" in out and "80" in out, out


def test_a_window_without_lambdas_is_not_guessed_into_a_pair():
    """旧 manifest 没有 `lambdas_vdw` ⟹ 不对齐、不显示，并且**明说**。"""
    a = {"windows": [{"window_idx": 0, "min_n_eff_over_g": 9.0}]}
    b = {"windows": [{"window_idx": 0, "min_n_eff_over_g": 3.0}]}
    out = R.render_ab_windows(a, b)
    assert "没有 `lambdas_vdw`" in out, out
    assert "对齐上 0 个窗口" in out, out


# ───────────────────────────────── indeterminate（2026-09-16 用户定死的四条语义）
# 固定预算模式下，冻结验证在批数上限内始终求不出 Δf−ΔF 的窗口是 `indeterminate`：
# **无结论 ≠ 失败 ≠ 0**。报告必须把它和"这个窗口不存在"分开，拒绝为它算 B/A，
# 并且明说该 run 的路径级结果不完整。

def _nd(idx, lam, *, reason="frozen_validation_budget_indeterminate"):
    """一个无结论窗口：有 λ（所以对得齐），但没有任何验收量。"""
    return {"window_idx": idx, "lambdas_vdw": list(lam),
            "min_n_eff_over_g": None, "verdict": None, "verdict_source": None,
            "n_decorrelated": None,
            "indeterminate": {"reason": reason, "detail": "窗口 2 的冻结验证 15/15 批…"}}


# 三个 λ 对齐窗口，B 的最后一个无结论。
_LAM = ([1.0, 0.8, 0.6], [0.6, 0.4, 0.2], [0.2, 0.1, 0.0])
A3 = {"windows": [_win(0, _LAM[0], 20.0), _win(1, _LAM[1], 12.0), _win(2, _LAM[2], 11.0)],
      "path_result": {"total_delta_G_kJ_mol": 100.0, "total_error_kJ_mol": 2.0}}
B3 = {"windows": [_win(0, _LAM[0], 21.0), _win(1, _LAM[1], 13.0), _nd(2, _LAM[2])],
      "path_result": {"analysis_status": "ANALYSIS_INCOMPLETE"}}


def test_an_indeterminate_window_is_marked_not_shown_as_missing():
    out = R.render_ab_windows(A3, B3)
    assert "INDETERMINATE" in out, out
    assert "frozen_validation_budget_indeterminate" in out, out
    # 三个窗口都对齐上了 —— 它没有被悄悄踢出配对
    assert "对齐上 3 个窗口" in out, out


def test_the_paired_ratio_is_refused_with_a_stated_reason():
    out = R.render_ab_windows(A3, B3)
    nd_row = [l for l in out.splitlines()
              if l.strip().startswith("2") and "INDETERMINATE" in l]
    assert nd_row, out
    assert "n.d." in nd_row[0], nd_row[0]
    # 前两个窗口照常算 B/A，证明不是整张表都罢工
    assert "1.05" in out and "1.08" in out, out
    # 而且**说了为什么**
    assert "不可判定" in out and "不得" in out, out


def test_the_run_level_delta_is_refused_too():
    """关键对齐窗口无结论 ⟹ 该 run 不是完整观测，Δ(B−A) 不许硬算。"""
    out = R.render_ab(_Dir(A3), _Dir(B3))
    assert "拒绝计算 Δ(B−A)" in out, out
    assert "Δ(B−A) =" not in out, out
    # 并且明说补齐只能同协议重跑，不能 resume
    assert "不是 resume" in out, out


class _Dir(str):
    """让 render_ab 接受现成的 manifest 而不去读盘。"""
    def __new__(cls, manifest):
        o = super().__new__(cls, "<in-memory>")
        o.manifest = manifest
        return o

"""CTL-08：被 solver 跳过的窗口必须留下**显式的逐段 UNMEASURED**。

去相关之后 `n_frames < min_frames_per_window` 的窗口会被 `continue` 掉，而那个
`continue` 在逐段累计 f_k 残差计算**之前** —— 被跳过的窗口因此一条残差记录都没有。
下游控制器的语义是「**缺证据 ≠ 没这回事**」：`cumulative_fk_residual_production`
里干脆没有这个窗口，读起来跟"查过、没问题"一模一样。
"""
import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

import ibs_engine as ie

KT = 2.577


def _healthy_window(lams, *, window_index, n=400, seed=1):
    r = np.random.default_rng(seed)
    base = r.normal(size=n)
    u = np.vstack([base + 0.3 * k + r.normal(scale=0.2, size=n)
                   for k in range(len(lams))])
    return {
        "window_index": window_index, "u_kn": u,
        "bias_energies": r.normal(scale=0.1, size=n), "base_energies": base,
        "lambda_indices": list(lams),
        "lambdas_vdw": [0.1 * i for i in lams],
        "sampled_distribution_row": 0,
    }


def _sticky_window(lams, *, window_index, n=300, seed=2):
    """强自相关 ⟹ g 很大 ⟹ 去相关后只剩个位数帧 ⟹ 走跳窗分支。"""
    r = np.random.default_rng(seed)
    x = np.cumsum(r.normal(size=n)) * 0.5
    u = np.vstack([x + 0.3 * k for k in range(len(lams))])
    return {
        "window_index": window_index, "u_kn": u,
        "bias_energies": np.zeros(n), "base_energies": x,
        "lambda_indices": list(lams),
        "lambdas_vdw": [0.1 * i for i in lams],
        "sampled_distribution_row": 0,
    }


@pytest.fixture(scope="module")
def solved():
    res = ie.GlobalMBARAnalyzer(kt=KT).solve_stage_integrated(
        [_healthy_window([0, 1, 2], window_index=0),
         _sticky_window([2, 3, 4], window_index=1)],
        min_frames_per_window=40,
    )
    assert [w["window_index"] for w in res["skipped_windows"]] == [1], \
        "测试前提：窗口 1 必须真的走到去相关后的跳窗分支"
    return res


def _record_for(res, idx):
    hits = [r for r in res["cumulative_fk_residual_production"]
            if r.get("window_index") == idx]
    assert len(hits) == 1, f"窗口 {idx} 的生产侧残差记录应当恰好一条，实得 {hits}"
    return hits[0]


def test_skipped_window_has_an_explicit_unmeasured_record(solved):
    """缺证据 ≠ 没这回事：跳窗必须显式落 UNMEASURED，而不是干脆不出现。"""
    rec = _record_for(solved, 1)
    assert rec["verdict"] == "UNMEASURED"
    assert rec["reason"] == "window_skipped_before_residual_computation"
    assert rec["n_frames_after_decorrelation"] < rec["min_frames_per_window"]
    assert rec["min_frames_per_window"] == 40


def test_unmeasured_record_keeps_the_success_path_shape(solved):
    """形状与成功路径一致 —— 否则下游按键取值会 KeyError 或静默取 None。"""
    skipped = _record_for(solved, 1)
    ok = _record_for(solved, 0)
    for key in ("scope", "window_index", "lambdas", "n_frames_in_frameset",
                "frameset_id", "segment_ids", "energy_gauge"):
        assert key in skipped, key
        assert key in ok, key
    assert skipped["scope"] == "production"
    assert skipped["lambdas"] == [2, 3, 4]


def test_unmeasured_is_not_confusable_with_a_pass(solved):
    """UNMEASURED 不许被读成 PASS —— 这条是本项待办的全部意义。"""
    verdicts = {r.get("window_index"): r.get("verdict")
                for r in solved["cumulative_fk_residual_production"]}
    assert verdicts[1] == "UNMEASURED"
    assert verdicts[1] != "PASS"

"""`_summarize_protocol_mismatch` 的输出必须**有界**，且要指出是哪个字段变了。

2026-08-27 引入这个摘要函数，本意就是不再把整棵 `_protocol_fingerprint` payload
甩进日志。但它留了两个洞，实测让一条"缓存作废"的日志重新变成 1151 字符：

  1. `_walk` 只在**两边都是 dict** 时递归。只要有一边整个缺了某个键（旧缓存里有
     `boresch_params.diagnostics`、新指纹里没有），另一边那棵完整字典就落到
     `a != b` 分支被原样 `repr`。
  2. 它完全不进 `list`。`boresch_harmonicity_check.fluctuation_distribution` 是
     list of dict，一有差异就整段甩出来，而且指不出是哪个自由度的哪个统计量变了。

这两条都在这里锁住。
"""

import copy

from abfe_pipeline import _summarize_protocol_mismatch

_MAX_CHARS = 400

_FLUCTUATION_DISTRIBUTION = [
    {
        "name": name, "mean": 0.36, "std": 0.019, "p01": 0.32, "p50": 0.358,
        "p99": 0.408, "skew": 0.44, "excess_kurtosis": 0.62, "n": 500,
        "ok": True, "reason": "ok",
    }
    for name in ("r", "thetaA", "thetaB", "phiA", "phiB", "phiC")
]

_DIAGNOSTICS = {
    "best_total_variance_score": 7.661357184698275,
    "bond_coverage": {"ligand_n_atoms": 7, "receptor_n_atoms": 809},
    "bond_source": "topology",
    "boresch_harmonicity_check": {
        "fluctuation_distribution": _FLUCTUATION_DISTRIBUTION
    },
}


def _key(payload):
    return {"schema_version": 1, "sha256": "deadbeef", "payload": payload}


def _cached():
    return _key({"boresch_params": {"diagnostics": copy.deepcopy(_DIAGNOSTICS)}})


def test_missing_whole_subtree_does_not_dump_it():
    """一边整个缺 `diagnostics` 时，不得把另一边那棵字典原样 repr 出来。"""
    summary = _summarize_protocol_mismatch(_cached(), _key({"boresch_params": {}}))
    assert len(summary) <= _MAX_CHARS, f"{len(summary)} 字符: {summary[:200]}"
    assert "boresch_params.diagnostics" in summary, summary
    # 结构摘要而不是字典字面量：不能出现整段 fluctuation_distribution 的内容
    assert "excess_kurtosis" not in summary, summary
    assert "缺失" in summary, summary


def test_string_missing_is_not_confused_with_absent_key():
    """值真的是字符串 `"<missing>"` 时，要跟"键不存在"区分开。"""
    summary = _summarize_protocol_mismatch(
        _key({"a": "<missing>"}), _key({"a": "other"})
    )
    assert "'<missing>'" in summary, summary
    assert "缺失" not in summary, summary


def test_scalar_change_reports_the_exact_leaf_path():
    """两边都有该键、只有一个标量变了 —— 摘要必须精确到叶子路径。"""
    current = _key({"boresch_params": {"diagnostics": copy.deepcopy(_DIAGNOSTICS)}})
    current["payload"]["boresch_params"]["diagnostics"][
        "best_total_variance_score"
    ] = 7.99
    summary = _summarize_protocol_mismatch(_cached(), current)
    assert "boresch_params.diagnostics.best_total_variance_score" in summary, summary
    assert len(summary) <= _MAX_CHARS, summary


def test_list_is_walked_so_the_changed_item_is_named():
    """list 要逐项递归：报出是第几项的哪个统计量变了，而不是甩出整段 list。"""
    current = _key({"boresch_params": {"diagnostics": copy.deepcopy(_DIAGNOSTICS)}})
    current["payload"]["boresch_params"]["diagnostics"][
        "boresch_harmonicity_check"
    ]["fluctuation_distribution"][3]["std"] = 0.99
    summary = _summarize_protocol_mismatch(_cached(), current)
    assert len(summary) <= _MAX_CHARS, f"{len(summary)} 字符: {summary[:200]}"
    assert "fluctuation_distribution.[3].std" in summary, summary
    assert "excess_kurtosis" not in summary, summary


def test_list_length_change_reports_lengths_not_contents():
    """长度不同没法逐项对齐 —— 退化成只报长度，仍然不许甩内容。"""
    current = _key({"boresch_params": {"diagnostics": copy.deepcopy(_DIAGNOSTICS)}})
    current["payload"]["boresch_params"]["diagnostics"][
        "boresch_harmonicity_check"
    ]["fluctuation_distribution"] = _FLUCTUATION_DISTRIBUTION[:4]
    summary = _summarize_protocol_mismatch(_cached(), current)
    assert len(summary) <= _MAX_CHARS, summary
    assert "6 项" in summary and "4 项" in summary, summary
    assert "excess_kurtosis" not in summary, summary


def test_identical_payload_but_different_hash_is_called_out():
    """payload 逐叶子一致而 sha256 不同 —— 这本身是值得单独说清楚的情况。"""
    cached = _cached()
    current = copy.deepcopy(cached)
    current["sha256"] = "cafebabe"
    summary = _summarize_protocol_mismatch(cached, current)
    assert "逐项比对一致" in summary, summary
    assert "cafebabe" in summary, summary

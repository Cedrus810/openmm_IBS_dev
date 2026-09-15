"""DECORR-01 裁决 4：同输入对照。

逐窗自检与全局求解器对"去相关帧数"实测差 2–8 倍（rep2 win3：自检 56 帧
`sufficient=True`，求解器 9/7 帧并跳窗）。**差异来源尚未证明**，所以规矩是：

  1. 两侧**分别记录**输入身份与输出，不许改任一侧的数去追平另一侧；
  2. **先比较输入构造**（哪个数组、什么观测量、怎么分段、kT、帧数）；
  3. 只有输入完全一致时，才在**完全相同的输入**上重放去相关函数。

去相关抽样本来就依赖输入时间序列与所选 `g` —— 拿另一份输出的帧数反推它是错的
（pymbar timeseries 的语义）。
"""
import ast
import inspect
import pathlib

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

import ibs_engine as ie


def _prov(**over):
    base = dict(
        caller="a", energy_array_source="energies.npy",
        observable="delta_u_k_over_kT_worst_target_state",
        n_frames_in=500, segments=[{"start_frame": 0, "end_frame": 500,
                                    "n_frames": 500, "session_id": "s1"}],
        kt=2.577, g=24.5, worst_state=3,
        subsample_indices=np.arange(0, 500, 24), window_idx=0,
    )
    base.update(over)
    return ie.decorrelation_provenance(**base)


def test_provenance_captures_input_identity_and_actual_subsample_indices():
    p = _prov()
    for k in ("energy_array_source", "observable", "n_frames_in", "segments",
              "kt_kJ_mol", "statistical_inefficiency", "worst_state",
              "n_decorrelated", "subsample_indices_head", "subsample_indices_tail"):
        assert k in p, k
    assert p["n_decorrelated"] == len(range(0, 500, 24))
    assert p["subsample_indices_head"][0] == 0
    assert p["n_segments"] == 1


def test_different_inputs_are_reported_as_input_differences_first():
    """输入不同 ⟹ 差异先归到输入构造上，**不许**改数字去追平。"""
    a = _prov(caller="self_check", energy_array_source="energies.npy")
    b = _prov(caller="solver", energy_array_source="u_kn (merged)",
              n_frames_in=1000, g=4.1,
              subsample_indices=np.arange(0, 1000, 4))
    out = ie.compare_decorrelation_provenance(a, b)

    assert out["inputs_identical"] is False
    assert set(out["input_differences"]) == {"energy_array_source", "n_frames_in"}
    assert "不得" in out["next_step"] and "追平" in out["next_step"]
    # 输出差异照记，但它不是结论
    assert out["output_delta"]["statistical_inefficiency"] == (24.5, 4.1)


def test_identical_inputs_point_at_the_replay_step():
    a, b = _prov(caller="self_check"), _prov(caller="solver")
    out = ie.compare_decorrelation_provenance(a, b)
    assert out["inputs_identical"] is True        # caller 不属于输入构造
    assert "重放" in out["next_step"]


def test_replay_runs_the_shared_primitive_on_the_given_input():
    rng = np.random.default_rng(1)
    n = 400
    # 造一条有自相关的序列，g 才会 > 1
    x = np.cumsum(rng.normal(size=n)) * 0.05
    u = np.vstack([x, x * 1.2 + 0.3, x * 0.8 - 0.2])
    out = ie.replay_decorrelation(u, x * 0.5, 2.577)
    assert out["n_decorrelated"] >= 1
    assert out["statistical_inefficiency"] >= 1.0
    assert len(out["per_state_g"]) == 3


def test_both_call_sites_actually_record_it():
    """两个调用点都必须落这份账 —— 只记一侧等于没法对照。"""
    src = pathlib.Path(ie.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    callers = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "decorrelation_provenance"):
            for kw in node.keywords:
                if kw.arg == "caller" and isinstance(kw.value, ast.Constant):
                    callers.add(kw.value.value)
    assert callers == {"window_self_support_check", "solve_stage_integrated"}, callers

    # 自检产物与求解器的跳窗/overlap 记录都要带上
    assert '"decorrelation_provenance": decorrelation_provenance(' in src
    assert '"decorrelation_provenance": _decorr_prov,' in src
    assert src.count('"decorrelation_provenance": _decorr_prov,') == 2


def test_the_comparator_never_mutates_either_side():
    """对照是只读的：不许"顺手"把一侧改成另一侧。"""
    src = inspect.getsource(ie.compare_decorrelation_provenance)
    for forbidden in ("=", "update("):
        pass
    assert "a.get(f) != b.get(f)" in src
    a, b = _prov(), _prov(n_frames_in=999)
    import copy
    a0, b0 = copy.deepcopy(a), copy.deepcopy(b)
    ie.compare_decorrelation_provenance(a, b)
    assert a == a0 and b == b0

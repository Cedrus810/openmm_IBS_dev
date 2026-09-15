"""CTL-09：DECORR-01 的同输入对照 —— 内容指纹、判定口径、生产落盘。

逐窗自检与全局求解器对"去相关帧数"实测差 2–8 倍。规矩没变：两侧分别记录、
先比输入构造、只有输入一致才在**完全相同的输入**上重放，**绝不**改任一侧的数
去追平另一侧。本文件钉死三件事：

  1. provenance 带**内容指纹**（光有"来源名称"字符串比不出东西）；
  2. 判定只看内容，名称不同但内容相同 ⟹ 输入相同；名称相同但内容不同 ⟹ 输入不同；
  3. 生产流程**自动**跑一次对照并落进该窗口的诊断记录，且失败不中断分析。
"""
import json

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

import ibs_engine as ie

KT = 2.577


def _prov(u, bias, **over):
    kw = dict(
        caller="a", energy_array_source="energies.npy",
        observable="delta_u_k_over_kT_worst_target_state",
        n_frames_in=int(np.asarray(bias).size), segments=None, kt=KT,
        g=24.5, worst_state=1, subsample_indices=np.arange(0, 40),
        window_idx=0, energy_array=u, bias_array=bias,
    )
    kw.update(over)
    return ie.decorrelation_provenance(**kw)


# ---------------------------------------------------------------- 1. 内容指纹
def test_provenance_carries_content_fingerprints_not_just_a_name():
    u = np.arange(30, dtype=float).reshape(3, 10)
    bias = np.linspace(0.0, 1.0, 10)
    p = _prov(u, bias)
    fp = p["energy_content_fingerprint"]
    assert fp["shape"] == [3, 10]
    assert len(fp["sha256"]) == 64
    assert fp["min"] == 0.0 and fp["max"] == 29.0
    assert p["bias_content_fingerprint"]["shape"] == [10]


def test_content_fingerprint_actually_tracks_content():
    u = np.arange(30, dtype=float).reshape(3, 10)
    bias = np.zeros(10)
    a = _prov(u, bias)["energy_content_fingerprint"]
    b = _prov(u + 1.0, bias)["energy_content_fingerprint"]
    assert a["sha256"] != b["sha256"]


# ------------------------------------------------- 2. 判定只看内容，不看名称
def test_different_source_names_with_identical_content_count_as_same_input():
    """自检写 `energies.npy`、求解器写 `u_kn` —— 名字不同，数是同一批。

    以前拿名称当判据 ⟹ 对照永远停在"输入不同"，重放那一步一次都跑不到。
    """
    u = np.arange(40, dtype=float).reshape(2, 20)
    bias = np.linspace(-1.0, 1.0, 20)
    a = _prov(u, bias, caller="window_self_support_check",
              energy_array_source="dual_window_3_vdw_energies.npy")
    b = _prov(u, bias, caller="solve_stage_integrated",
              energy_array_source="window_outputs[...]['u_kn']")
    out = ie.compare_decorrelation_provenance(a, b)

    assert out["inputs_identical"] is True
    assert out["content_fingerprints_available"] is True
    # 名称差异照列出来当溯源线索，但**不进**判定
    assert "energy_array_source" in out["name_only_differences"]
    assert "energy_array_source" not in out["decisive_input_differences"]
    assert "重放" in out["next_step"]


def test_same_source_name_with_different_content_counts_as_different_input():
    u = np.arange(40, dtype=float).reshape(2, 20)
    a = _prov(u, np.zeros(20), energy_array_source="energies.npy")
    b = _prov(u * 2.0, np.zeros(20), energy_array_source="energies.npy")
    out = ie.compare_decorrelation_provenance(a, b)

    assert out["inputs_identical"] is False
    assert "energy_content_fingerprint" in out["decisive_input_differences"]
    assert not out["name_only_differences"]
    assert "不得" in out["next_step"]


def test_missing_fingerprints_are_reported_as_no_evidence():
    """两侧都没指纹 = 没有证据，下游不许读成"测过且一致"。"""
    old = ie.decorrelation_provenance(
        caller="x", energy_array_source="energies.npy",
        observable="o", n_frames_in=10, segments=None, kt=KT, g=1.0,
        worst_state=0, subsample_indices=np.arange(10), window_idx=0,
    )
    out = ie.compare_decorrelation_provenance(old, dict(old))
    assert out["content_fingerprints_available"] is False


# --------------------------------------------------------- 3. 生产流程自动跑
def _window(lams, *, window_index, n=400, seed=1):
    r = np.random.default_rng(seed)
    base = r.normal(size=n)
    u = np.vstack([base + 0.3 * k + r.normal(scale=0.2, size=n)
                   for k in range(len(lams))])
    return {
        "window_index": window_index, "u_kn": u,
        "bias_energies": r.normal(scale=0.1, size=n), "base_energies": base,
        "lambda_indices": list(lams), "lambdas_vdw": [0.1 * i for i in lams],
        "sampled_distribution_row": 0,
    }


def _write_self_support(tmp_path, w, name, **over):
    """把"自检那一侧"的 provenance 落成真实产物文件。"""
    path = tmp_path / name
    prov = ie.decorrelation_provenance(
        caller="window_self_support_check",
        energy_array_source="dual_window_0_vdw_energies.npy",
        observable="delta_u_k_over_kT_worst_target_state",
        n_frames_in=int(np.asarray(w["bias_energies"]).size),
        segments=None, kt=KT, g=56.0, worst_state=0,
        subsample_indices=np.arange(56), window_idx=int(w["window_index"]),
        energy_array=over.pop("energy_array", w["u_kn"]),
        bias_array=w["bias_energies"],
    )
    path.write_text(json.dumps({"decorrelation_provenance": prov}),
                    encoding="utf-8")
    w["self_support_path"] = str(path)
    return path


def _solve(windows):
    return ie.GlobalMBARAnalyzer(kt=KT).solve_stage_integrated(
        windows, min_frames_per_window=10
    )


def _cross_check(res, idx):
    hits = [r for r in res["window_overlap_diagnostics"]
            if r.get("window_index") == idx]
    assert len(hits) == 1
    return hits[0]["decorrelation_cross_check"]


def test_solver_runs_the_cross_check_and_replays_on_identical_inputs(tmp_path):
    w0 = _window([0, 1, 2], window_index=0)
    w1 = _window([2, 3, 4], window_index=1, seed=7)
    _write_self_support(tmp_path, w0, "w0_self_support.json")
    res = _solve([w0, w1])

    xc = _cross_check(res, 0)
    assert xc is not None, "拿得到两份 provenance 就必须自动跑对照并落盘"
    assert xc["inputs_identical"] is True
    # 输入一致 ⟹ 在完全相同的输入上重放一次
    assert "replay" in xc
    assert xc["replay"]["n_decorrelated"] >= 1
    assert xc["replay"]["statistical_inefficiency"] >= 1.0

    # 没有自检产物的窗口：对照缺席、但分析照常
    assert _cross_check(res, 1) is None


def test_replay_is_skipped_when_inputs_differ(tmp_path):
    """输入不同还重放 ⟹ 又造一个互相不可比的第三个数。"""
    w0 = _window([0, 1, 2], window_index=0)
    w1 = _window([2, 3, 4], window_index=1, seed=7)
    _write_self_support(tmp_path, w0, "w0_self_support.json",
                        energy_array=w0["u_kn"] * 3.0)
    xc = _cross_check(_solve([w0, w1]), 0)

    assert xc["inputs_identical"] is False
    assert "replay" not in xc
    assert "energy_content_fingerprint" in xc["decisive_input_differences"]


def test_cross_check_failure_never_breaks_the_analysis(tmp_path, monkeypatch):
    w0 = _window([0, 1, 2], window_index=0)
    w1 = _window([2, 3, 4], window_index=1, seed=7)
    _write_self_support(tmp_path, w0, "w0_self_support.json")

    def boom(*_a, **_k):
        raise RuntimeError("对照炸了")

    monkeypatch.setattr(ie, "compare_decorrelation_provenance", boom)
    res = _solve([w0, w1])

    assert "对照炸了" in _cross_check(res, 0)["error"]
    # 主分析必须完好
    assert res["total_delta_G"] is not None
    assert len(res["window_overlap_diagnostics"]) == 2

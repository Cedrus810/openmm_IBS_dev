"""S2-A 逐段门：每个采样段绑**它自己的**冻结 f_k，各算各的残差。

先前多段窗口直接记 `no_effective_f_k_for_these_frames` ⟹ 这项证据对多段窗口
**永久缺失**（循环一旦用换 Epoch 修好一个窗口，它就再也拿不到）。
裁决 3 的口径：
  · 帧切片由 `sampling_source_id` 给出（有序帧映射，不是猜边界）；
  · 每段用 `sampling_segment_f_k[i]`，不许拿某一段的 f_k 冒充全体帧；
  · **绝不**拿合并后的 ΔF 作逐段望远镜相消；
  · 窗口级：任一有效段 FAIL 则 FAIL；否则有缺证据段为 UNMEASURED；全过才 PASS。
"""
import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

from ibs_engine import _per_segment_cumulative_fk_residual as PSEG


def _args(n_states=3, n_frames=40):
    rng = np.random.default_rng(0)
    sampling = rng.normal(size=(n_states, n_frames))
    bias = rng.normal(size=n_frames)
    base = rng.normal(size=n_frames)
    return sampling, bias, base, list(range(n_states))


def test_each_segment_is_solved_on_its_own_frames_with_its_own_f_k(monkeypatch):
    seen = []

    def fake(sampling_kj, bias_kj, base_kj, win_lams, kt, f_frozen, **kw):
        seen.append((sampling_kj.shape[1], float(f_frozen[0])))
        return {"error": None, "verdict": "PASS",
                "cumulative_residual_span_kJ_mol": 1.0,
                "cumulative_residual_span_sigma_kJ_mol": 0.1,
                "threshold_kJ_mol": 10.0}

    monkeypatch.setattr("ibs_engine.cumulative_fk_residual", fake)
    s, b, ba, lams = _args(n_frames=40)
    ids = np.array([0] * 25 + [1] * 15)
    out = PSEG(s, b, ba, lams, 2.5, source_ids=ids,
               segment_f_ks=[np.array([1.0, 2.0, 3.0]), np.array([9.0, 8.0, 7.0])],
               provenance=[{"segment_index": 1, "source_dir": "/r/vanishing"},
                           {"segment_index": 2, "source_dir": "/r/vanishing_2"}])

    # 每段只拿自己的帧、自己的 f_k —— 不是把两段混在一起解一次
    assert seen == [(25, 1.0), (15, 9.0)]
    assert out["per_segment_gate"] is True and out["n_segments"] == 2
    assert out["verdict"] == "PASS"
    # 身份、帧范围、结论可审计
    assert [x["source_dir"] for x in out["segments"]] == [
        "/r/vanishing", "/r/vanishing_2"]
    assert out["segments"][0]["frame_range"] == [0, 24]
    assert out["segments"][1]["frame_range"] == [25, 39]
    assert out["segments"][0]["n_frames"] == 25


@pytest.mark.parametrize("verdicts,expected", [
    (["PASS", "PASS"], "PASS"),
    (["PASS", "FAIL_CUMULATIVE_FK"], "FAIL_CUMULATIVE_FK"),
    (["FAIL_CUMULATIVE_FK", "UNMEASURED"], "FAIL_CUMULATIVE_FK"),
    (["PASS", "UNMEASURED"], "UNMEASURED"),
    (["UNMEASURED", "UNMEASURED"], "UNMEASURED"),
])
def test_window_level_verdict_composition(monkeypatch, verdicts, expected):
    it = iter(verdicts)

    def fake(*a, **kw):
        v = next(it)
        return ({"error": "not_enough_frames"} if v == "UNMEASURED"
                else {"error": None, "verdict": v,
                      "cumulative_residual_span_kJ_mol": 1.0})

    monkeypatch.setattr("ibs_engine.cumulative_fk_residual", fake)
    s, b, ba, lams = _args(n_frames=20)
    out = PSEG(s, b, ba, lams, 2.5,
               source_ids=np.array([0] * 10 + [1] * 10),
               segment_f_ks=[np.ones(3), np.ones(3)], provenance=None)
    assert out["verdict"] == expected


def test_a_segment_with_no_frames_is_unmeasured_not_a_pass(monkeypatch):
    monkeypatch.setattr(
        "ibs_engine.cumulative_fk_residual",
        lambda *a, **kw: {"error": None, "verdict": "PASS",
                          "cumulative_residual_span_kJ_mol": 1.0})
    s, b, ba, lams = _args(n_frames=20)
    out = PSEG(s, b, ba, lams, 2.5, source_ids=np.zeros(20, dtype=int),
               segment_f_ks=[np.ones(3), np.ones(3)], provenance=None)
    assert out["segments"][1]["verdict"] == "UNMEASURED"
    assert out["verdict"] == "UNMEASURED"          # 缺证据 ≠ 通过


def test_missing_inputs_fail_closed():
    s, b, ba, lams = _args()
    assert PSEG(s, b, ba, lams, 2.5, source_ids=None, segment_f_ks=[np.ones(3)],
                provenance=None)["error"] == "no_effective_f_k_for_these_frames"
    assert PSEG(s, b, ba, lams, 2.5, source_ids=np.zeros(3, dtype=int),
                segment_f_ks=[np.ones(3)], provenance=None
                )["error"] == "sampling_source_id_length_mismatch"


def test_no_cross_segment_telescoping_is_reported():
    """窗口级**不给**合成 span —— 逐段残差不与合并后的 ΔF 望远镜相消。"""
    import inspect
    src = inspect.getsource(PSEG)
    assert "max_segment_span_kJ_mol_REPORT_ONLY" in src
    assert "没有**跨段望远镜相消" in src

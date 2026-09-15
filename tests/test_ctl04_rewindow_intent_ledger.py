"""CTL-04 ③：rewindow 的**意图**必须先于采样落盘，恢复时按盘上产物核销。

原顺序是「算身份 → 采样(GPU) → 写台账 → 求解」。采样是整条链上唯一烧 GPU 的
一步，它一旦被打断（崩溃 / 被 kill / 节点掉线），台账里**没有这个 identity 的
任何记录** ⟹ 那些已经烧掉 GPU 的子系综目录成了孤儿：控制器只读台账，看不见
它们，下一跑会重新建一遍、白烧一次。

现在：采样前先写 `status="SAMPLING_INTENT"`，采样成功推进到 `"SAMPLED"`；
恢复时 `_reconcile_rewindow_intents()` 按产物核销（有产物 ⟹ SAMPLED，
零产物 ⟹ ABANDONED_NO_PRODUCT，**目录一律不删**）。
合并求解**只采信 SAMPLED**，且 `replaced` 集合同样只算 SAMPLED ——
否则一个其实没采出来的条目会把父窗骗成"已被取代"，产出截断的 ΔG。
"""
import json
import os
import pathlib
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abfe_pipeline import ABFEPipeline  # noqa: E402

R4 = [(0, 3), (3, 8), (8, 11), (11, 13)]     # 4 个物理窗口、13 个态
LAM = [1.0 - x / 12.0 for x in range(13)]


def _pipe():
    """只用到 `_log` 的最小 stub —— 别为了一个日志函数去建整条流水线。"""
    obj = object.__new__(ABFEPipeline)
    obj.logs = []
    obj.sampling_score_sha256 = None
    obj._log = obj.logs.append
    return obj


def _run(tmp_path):
    run = pathlib.Path(tmp_path) / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "vanishing").mkdir()
    return run


def _conv(d, li):
    (pathlib.Path(d) / f"dual_window_{li}_vdw_convergence.json").write_text("{}")


def _ledger(run):
    p = run / "checkpoints" / "stage2_rewindow_ledger.json"
    return json.loads(p.read_text()) if p.is_file() else {}


# --------------------------------------------------------------------------
# ① 采样前就得有账：run_once 炸掉也要留下 SAMPLING_INTENT
# --------------------------------------------------------------------------
def test_intent_is_persisted_before_sampling(tmp_path):
    run = _run(tmp_path)
    pipe = _pipe()

    def _boom(*a, **k):
        raise RuntimeError("节点掉线")

    with pytest.raises(RuntimeError, match="节点掉线"):
        pipe._immutable_rewindow_step(
            _boom, view={}, window_idx=1, lam=LAM, ranges=R4,
            stage_dir=str(run / "vanishing"),
            checkpoint_dir=str(run / "checkpoints"),
            base_unit=250000, kt=2.5,
        )

    led = _ledger(run)
    assert len(led) == 1, "采样被打断时台账必须已经有这一条（否则子系综成孤儿）"
    (entry,) = led.values()
    assert entry["status"] == "SAMPLING_INTENT"
    # 恢复时要靠这些字段找产物 / 重建覆盖，缺一不可
    assert entry["parent_window"] == 1 and entry["parent_range"] == [3, 8]
    assert entry["child_ranges"] and entry["output_dir"] and entry["checkpoint_dir"]
    assert isinstance(entry["solver_index_base"], int)


def test_status_advances_to_sampled_after_successful_sampling(tmp_path):
    run = _run(tmp_path)
    pipe = _pipe()
    seen = {}

    def _ok(*a, **k):
        # 采样这一刻，台账里必须已经是 INTENT（而不是"还没有这条"）
        seen["at_sampling"] = _ledger(run)
        return {"converged": True}

    pipe._solve_with_rewindow_children = lambda **k: {"converged": True}
    pipe._immutable_rewindow_step(
        _ok, view={}, window_idx=1, lam=LAM, ranges=R4,
        stage_dir=str(run / "vanishing"),
        checkpoint_dir=str(run / "checkpoints"),
        base_unit=250000, kt=2.5,
    )
    assert [e["status"] for e in seen["at_sampling"].values()] == ["SAMPLING_INTENT"]
    assert [e["status"] for e in _ledger(run).values()] == ["SAMPLED"]


# --------------------------------------------------------------------------
# ② 核销：按盘上产物判，绝不删目录
# --------------------------------------------------------------------------
def _put_intent(run, ident, *, parent, children, mk_products):
    out = run / f"vanishing_rewindow_{ident}"
    out.mkdir()
    for li in mk_products:
        _conv(out, li)
    led = _ledger(run)
    led[ident] = {
        "identity": ident, "path_version": 0, "solver_index_base": 10000,
        "parent_window": parent, "parent_range": list(R4[parent]),
        "child_ranges": [list(c) for c in children],
        "output_dir": str(out),
        "checkpoint_dir": str(run / "checkpoints" / f"rewindow_{ident}"),
        "f_k_scope": "own_frozen_f_k_per_child_ensemble",
        "blocks": [], "status": "SAMPLING_INTENT",
    }
    (run / "checkpoints" / "stage2_rewindow_ledger.json").write_text(json.dumps(led))
    return out


def test_reconcile_promotes_intent_with_products_and_abandons_empty_one(tmp_path):
    run = _run(tmp_path)
    done = _put_intent(run, "aaa111", parent=1, children=[[3, 6], [5, 8]],
                       mk_products=[0, 1])
    empty = _put_intent(run, "bbb222", parent=2, children=[[8, 10], [9, 11]],
                        mk_products=[])
    pipe = _pipe()
    pipe._reconcile_rewindow_intents(str(run / "checkpoints"))

    led = _ledger(run)
    assert led["aaa111"]["status"] == "SAMPLED"
    assert led["bbb222"]["status"] == "ABANDONED_NO_PRODUCT"
    # 本仓库规矩：不原地删实验产物
    assert done.is_dir() and empty.is_dir()
    assert any("SAMPLED" in m for m in pipe.logs)
    assert any("ABANDONED_NO_PRODUCT" in m for m in pipe.logs)


def test_reconcile_leaves_sampled_and_legacy_entries_alone(tmp_path):
    run = _run(tmp_path)
    _put_intent(run, "ccc333", parent=1, children=[[3, 6], [5, 8]], mk_products=[0])
    led = _ledger(run)
    led["ccc333"]["status"] = "SAMPLED"
    led["old999"] = dict(led["ccc333"], identity="old999", parent_window=2)
    del led["old999"]["status"]                      # 老台账：根本没有这个字段
    (run / "checkpoints" / "stage2_rewindow_ledger.json").write_text(json.dumps(led))

    _pipe()._reconcile_rewindow_intents(str(run / "checkpoints"))
    out = _ledger(run)
    assert out["ccc333"]["status"] == "SAMPLED"
    assert "status" not in out["old999"], "老条目不该被这次改动改写"


# --------------------------------------------------------------------------
# ③ 合并求解：只采信 SAMPLED，且 fail-closed 的 replaced 集合同口径
# --------------------------------------------------------------------------
def _wire_solver(pipe, monkeypatch, loaded):
    import ibs_engine
    monkeypatch.setattr(
        ibs_engine, "solve_stage_integrated",
        lambda **k: {"converged": True, "n_outputs": len(k["window_outputs"])},
        raising=False,
    )
    pipe._load_ibs_window_outputs_merged = lambda *a, **k: []
    def _from_dir(out_dir, children, *a, **k):
        loaded.append(out_dir)
        return [{"d": out_dir}] * len(children)
    pipe._load_ibs_window_outputs_from_dir = _from_dir


def test_abandoned_entry_does_not_count_as_replacing_its_parent(tmp_path, monkeypatch):
    """核心 fail-closed：没采出来的条目不得把父窗骗成"已被取代"。"""
    run = _run(tmp_path)
    for i in (0, 2, 3):                      # 父窗 1 的产物**不在**采样段里
        _conv(run / "vanishing", i)
    _put_intent(run, "bbb222", parent=1, children=[[3, 6], [5, 8]], mk_products=[])
    pipe = _pipe()
    loaded = []
    _wire_solver(pipe, monkeypatch, loaded)
    pipe._reconcile_rewindow_intents(str(run / "checkpoints"))

    with pytest.raises(RuntimeError, match="既不在任何采样段里"):
        pipe._solve_with_rewindow_children(
            stage_dir=str(run / "vanishing"),
            checkpoint_dir=str(run / "checkpoints"),
            ranges=R4, lam=LAM, kt=2.5,
        )
    assert loaded == [], "ABANDONED 的子系综不得进合并求解"


def test_sampled_and_legacy_entries_are_still_accepted(tmp_path, monkeypatch):
    run = _run(tmp_path)
    for i in (0, 2, 3):
        _conv(run / "vanishing", i)
    out = _put_intent(run, "aaa111", parent=1, children=[[3, 6], [5, 8]],
                      mk_products=[0, 1])
    led = _ledger(run)
    del led["aaa111"]["status"]              # 老台账条目：没有 status ⟹ 当 SAMPLED
    (run / "checkpoints" / "stage2_rewindow_ledger.json").write_text(json.dumps(led))

    pipe = _pipe()
    loaded = []
    _wire_solver(pipe, monkeypatch, loaded)
    res = pipe._solve_with_rewindow_children(
        stage_dir=str(run / "vanishing"),
        checkpoint_dir=str(run / "checkpoints"),
        ranges=R4, lam=LAM, kt=2.5,
    )
    assert res["n_outputs"] == 2 and loaded == [str(out)]

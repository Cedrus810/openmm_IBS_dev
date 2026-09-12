"""λ 路径版本记录的离线契约测试（不需要 GPU，不建任何 OpenMM Context）。

验收对齐设计的三条：插点后中断重启**路径不倒退、不重复插点、不因编号变化丢数据**。
"""

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

# _atomic_write 惰性 import ibs_engine（落盘要复用它那份 fsync + 目录同步的实现）。
pytest.importorskip("openmm")

import lambda_path_versions as lpv


# v1: 状态 0..9，窗口 [0,4) [4,8) [8,10)
LC_V1 = [0.0] * 10
LV_V1 = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0]
WR_V1 = [[0, 4], [4, 8], [8, 10]]

# v2: 在 0.5 与 0.4 之间插入 NEW=0.45，后面顺延；按设计里的例子重新分组
LV_V2 = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.3, 0.2, 0.0]
LC_V2 = [0.0] * 11
WR_V2 = [[0, 4], [4, 8], [8, 11]]

INSERT = {"interval_state_ids": ["a", "b"], "inserted_lambda_vdw": 0.45}


def _init(tmp_path):
    return lpv.init_version(str(tmp_path), LC_V1, LV_V1, WR_V1)


def _append(tmp_path, detail=None):
    return lpv.append_version(
        str(tmp_path), LC_V2, LV_V2, WR_V2,
        kind="insert_lambda", reason="f_k_not_converged",
        detail=INSERT if detail is None else detail,
    )


def test_init_creates_v1_and_points_at_it(tmp_path):
    record = _init(tmp_path)
    assert record["version"] == 1
    assert record["parent_version"] is None
    assert len(record["states"]) == 10
    assert lpv.load_current(str(tmp_path))["version"] == 1


def test_init_is_idempotent_and_never_rewinds_an_evolved_path(tmp_path):
    """resume 不能重新预优化把已经演化到 v2 的路径打回 v1。"""
    _init(tmp_path)
    _append(tmp_path)
    again = lpv.init_version(str(tmp_path), LC_V1, LV_V1, WR_V1)
    assert again["version"] == 2, "已有 v2 时 init 必须原样交回 v2"
    assert lpv.load_current(str(tmp_path))["version"] == 2


def test_state_identity_survives_renumbering(tmp_path):
    """old7 后移一位仍是同一个状态 —— 身份由 λ 决定，不由位置决定。"""
    v1 = _init(tmp_path)
    v2 = _append(tmp_path)
    old = {s["id"]: i for i, s in enumerate(v1["states"])}
    new = {s["id"]: i for i, s in enumerate(v2["states"])}
    moved = lpv.state_id(0.0, 0.4)
    assert old[moved] == 6 and new[moved] == 7, "λ=0.4 应从第 7 位移到第 8 位"
    assert set(old) < set(new), "v1 的每个状态都必须还在 v2 里"
    assert set(new) - set(old) == {lpv.state_id(0.0, 0.45)}


def test_changed_windows_reuses_prefix_and_flags_the_rest(tmp_path):
    v1 = _init(tmp_path)
    v2 = _append(tmp_path)
    diff = lpv.changed_windows(v1, v2)
    assert {r["new_index"] for r in diff["reusable"]} == {0}, "插点之前的窗口应可复用"
    assert diff["changed"] == [1, 2]
    assert diff["dropped_old_indices"] == [1, 2]


def test_changed_windows_matches_by_content_not_index(tmp_path):
    """组成不变、只是编号后移的窗口算可复用（产物改名即可）。"""
    v1 = _init(tmp_path)
    shifted = json.loads(json.dumps(v1))
    shifted["windows"] = [
        {"index": w["index"] + 1, "state_ids": w["state_ids"]} for w in v1["windows"]
    ]
    diff = lpv.changed_windows(v1, shifted)
    assert diff["changed"] == []
    assert [(r["old_index"], r["new_index"]) for r in diff["reusable"]] == [(0, 1), (1, 2), (2, 3)]


def test_same_insert_event_does_not_insert_twice(tmp_path):
    _init(tmp_path)
    first = _append(tmp_path)
    second = _append(tmp_path)
    assert second["version"] == first["version"] == 2
    assert lpv.existing_versions(str(tmp_path)) == [1, 2]


def test_orphan_version_after_crash_is_adopted_not_reinserted(tmp_path):
    """写完 v2 但还没推指针就被杀：重启后必须采纳 v2，而不是插成 v3。"""
    _init(tmp_path)
    _append(tmp_path)
    # 把指针手工退回 v1，模拟"v2 已落盘、指针没来得及写"
    lpv._atomic_write(lpv.pointer_path(str(tmp_path)),
                      {"version": 1, "protocol_version": lpv.PATH_VERSION_PROTOCOL_VERSION})
    assert lpv.load_current(str(tmp_path))["version"] == 1

    adopted = _append(tmp_path)
    assert adopted["version"] == 2, "同一个插点事件必须采纳孤儿 v2"
    assert lpv.existing_versions(str(tmp_path)) == [1, 2], "不得产生 v3"
    assert lpv.load_current(str(tmp_path))["version"] == 2


def test_a_genuinely_different_insert_creates_a_new_version(tmp_path):
    _init(tmp_path)
    _append(tmp_path)
    third = _append(tmp_path, detail={"interval_state_ids": ["c", "d"],
                                      "inserted_lambda_vdw": 0.45})
    assert third["version"] == 3 and third["parent_version"] == 2


def test_pointer_never_rewinds(tmp_path):
    _init(tmp_path)
    _append(tmp_path)
    with pytest.raises(ValueError, match="倒退"):
        lpv._publish(str(tmp_path), 1)


def test_old_versions_stay_immutable(tmp_path):
    v1 = _init(tmp_path)
    before = Path(lpv.version_path(str(tmp_path), 1)).read_bytes()
    _append(tmp_path)
    assert Path(lpv.version_path(str(tmp_path), 1)).read_bytes() == before
    assert lpv.load_version(str(tmp_path), 1)["content_sha256"] == v1["content_sha256"]


def test_tampered_record_is_rejected_fail_closed(tmp_path):
    _init(tmp_path)
    path = lpv.version_path(str(tmp_path), 1)
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    record["window_ranges"] = [[0, 10]]          # 改内容但不改 sha256
    Path(path).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="content_sha256"):
        lpv.load_version(str(tmp_path), 1)


def test_missing_pointer_reads_as_no_path(tmp_path):
    _init(tmp_path)
    os.remove(lpv.pointer_path(str(tmp_path)))
    assert lpv.load_current(str(tmp_path)) is None


def test_history_walks_parents_in_order(tmp_path):
    _init(tmp_path)
    _append(tmp_path)
    assert [r["version"] for r in lpv.history(str(tmp_path))] == [1, 2]


def test_window_ranges_must_cover_every_state(tmp_path):
    with pytest.raises(ValueError, match="未完整覆盖"):
        lpv.init_version(str(tmp_path), LC_V1, LV_V1, [[0, 4], [4, 8]])


def test_duplicate_lambda_states_are_rejected(tmp_path):
    lv = list(LV_V1)
    lv[5] = lv[4]
    with pytest.raises(ValueError, match="重复状态"):
        lpv.init_version(str(tmp_path), LC_V1, lv, WR_V1)


def test_append_requires_an_existing_current_version(tmp_path):
    with pytest.raises(ValueError, match="init_version"):
        _append(tmp_path)


def test_state_id_is_stable_under_float_noise(tmp_path):
    assert lpv.state_id(0.0, 0.4) == lpv.state_id(0.0, 0.4 + 1e-12)
    assert lpv.state_id(0.0, -0.0) == lpv.state_id(0.0, 0.0)

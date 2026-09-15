"""CTL-05：rewindow 台账必须按**当前 `path_version`** 隔离。

路径演化（插 λ / 拆末窗）之后，旧布局下建的子系综描述的是**另一套 λ 区间**：
  · 拿它们参与当前求解 ⟹ 混进不可比的帧；
  · 旧的 `parents_done` 还会把新的修补挡在门外（"这个父窗已经建过子系综了"
    —— 那是上一条布局的事）。
"""
import json
import os
import pathlib
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abfe_preoptimizer import Stage2RepairController  # noqa: E402

from test_rewindow_sampling_units import _with_rewindow  # noqa: E402


def _set_entry_version(run, version):
    f = pathlib.Path(run) / "checkpoints" / "stage2_rewindow_ledger.json"
    d = json.loads(f.read_text())
    d["abc123"]["path_version"] = version
    f.write_text(json.dumps(d))


def test_units_from_another_path_version_are_excluded(tmp_path):
    run = _with_rewindow(tmp_path, child_states=[("ANALYSIS_ELIGIBLE", 250000)])
    v0 = Stage2RepairController(run, "vanishing").read()
    assert len(v0["sampling_units"]) == 2          # 当前布局（v1）下建的两个子窗

    _set_entry_version(run, 7)                     # 改成另一条布局
    v1 = Stage2RepairController(run, "vanishing").read()
    assert v1["sampling_units"] == [], "旧布局的子窗混进了当前视图"
    assert v1["immutable_rewindow"]["parents_done"] == [], (
        "旧布局的 parents_done 会把新修补挡在门外"
    )
    assert "abc123" in v1["immutable_rewindow"]["entries_from_other_path_versions"]


def test_entries_without_a_version_are_still_accepted(tmp_path):
    """老台账没有这个字段 ⟹ 按"无法判定版本"放行，别把既有 run 判废。"""
    run = _with_rewindow(tmp_path, child_states=[("ANALYSIS_ELIGIBLE", 250000)])
    f = pathlib.Path(run) / "checkpoints" / "stage2_rewindow_ledger.json"
    d = json.loads(f.read_text())
    d["abc123"].pop("path_version", None)
    f.write_text(json.dumps(d))

    v = Stage2RepairController(run, "vanishing").read()
    assert len(v["sampling_units"]) == 2
    assert v["immutable_rewindow"]["parents_done"] == [1]


def test_a_parent_rewindowed_under_an_old_layout_can_be_rewindowed_again(tmp_path):
    """`parents_done` 只算当前布局 —— 否则新布局下这个父窗永远修不了。"""
    run = _with_rewindow(tmp_path, child_states=[("ANALYSIS_ELIGIBLE", 250000)])
    _set_entry_version(run, 7)
    v = Stage2RepairController(run, "vanishing").read()
    assert 1 not in (v["immutable_rewindow"]["parents_done"] or [])

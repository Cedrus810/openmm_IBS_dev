"""CTL-01：控制器读哪一份 stage 结果，**不能靠文件名顺序决定**。

`.`(46) < `_`(95) ⟹ `stage2_vanishing.json`（stage 缓存完成标记，可能是几轮之前
写的）字典序**永远排在** `stage2_vanishing_autonomous_inprogress.json 之前。
先前 `sorted(glob("stage2_*.json"))` 取第一个带 `converged` 的 ⟹ 自治循环每轮
刚写出的新结果根本读不到，子窗的 solver 帧数与跳窗状态全从过期结果里读，
11/20 的新证据进不了 `decide()`。
"""
import json
import os
import pathlib
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abfe_preoptimizer import Stage2RepairController  # noqa: E402

from test_stage2_repair_controller import _mkrun, R4, FULL  # noqa: E402

# 🔑 [2026-09-15] 下面的 fixture 一律用 `analysis_status` 而不是 `converged`：
# `_read_stage_result()` 的**嗅探键**同日从 `converged` 换成了
# `analysis_status` / `total_delta_G`（`converged` 是删键不是改义）。写老键的
# fixture 根本不会被认成一份 stage 结果 ⟹ 返回 None ⟹ 本文件测的"选哪一份"
# 一行都执行不到。取值一律 `ANALYSIS_INCOMPLETE`：本文件测的是**选择顺序与
# 布局隔离**，跟这份结果通没通过无关，给 COMPLETE 反而会引入无关语义。


def _run(tmp_path, stale=None, fresh=None):
    run = _mkrun(tmp_path, windows=FULL, ranges=R4, n_states=13,
                 stage_result=stale)
    if fresh is not None:
        (pathlib.Path(run) / "checkpoints"
         / "stage2_vanishing_autonomous_inprogress.json").write_text(
            json.dumps(fresh))
    return run


def test_the_inprogress_result_wins_over_the_stale_completion_marker(tmp_path):
    run = _run(tmp_path,
               stale={"analysis_status": "ANALYSIS_INCOMPLETE",
                      "min_overlap": 0.001, "tag": "stale"},
               fresh={"analysis_status": "ANALYSIS_INCOMPLETE",
                      "min_overlap": 0.42, "tag": "fresh"})
    got = Stage2RepairController(run, "vanishing")._read_stage_result()
    assert got["tag"] == "fresh", "又按文件名顺序挑了那份过期的完成标记"


def test_a_result_from_another_path_version_is_skipped(tmp_path):
    """带 `path_version` 且与当前布局不符 ⟹ 那是**另一个量**，不是"旧一点"。"""
    run = _run(tmp_path,
               stale={"analysis_status": "ANALYSIS_INCOMPLETE",
                      "tag": "v1", "path_version": 1},
               fresh={"analysis_status": "ANALYSIS_INCOMPLETE",
                      "tag": "other", "path_version": 7})
    got = Stage2RepairController(run, "vanishing")._read_stage_result()
    assert got["tag"] == "v1", "混进了另一条布局的结论"


def test_old_artifacts_without_a_version_are_still_accepted(tmp_path):
    """不带该字段的老产物按"无法判定版本"放行，不制造新的 fail。"""
    run = _run(tmp_path, stale={"analysis_status": "ANALYSIS_INCOMPLETE",
                                "tag": "legacy"})
    got = Stage2RepairController(run, "vanishing")._read_stage_result()
    assert got["tag"] == "legacy"


def test_the_writer_stamps_the_current_path_version():
    """写盘侧不盖版本号，读盘侧的隔离就形同虚设。"""
    import inspect

    import abfe_pipeline

    src = inspect.getsource(
        abfe_pipeline.ABFEPipeline._persist_inprogress_stage_result)
    assert 'payload["path_version"]' in src

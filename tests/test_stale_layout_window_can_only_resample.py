"""布局变过之后，旧产物**只能重采**，不能拿去重解。

真机 `cyclod_ligand1/rep2` 17:57 崩在这上面：

    [自治] 执行 PROBE_REANCHOR_EPOCH 失败：ValueError('窗口 3 状态数与 window_ranges 不符')
    [自治] 主循环异常：ValueError('窗口 3 状态数与 window_ranges 不符')

实测盘面：路径已演化到 v2（win3 从 6 态涨到 7 态），`ibs_state` 的
`lambdas_vdw`/`f_k` 已是 7，而该窗口的生产产物（`convergence.json` / `energies.npy`）
还是 **6 态的旧帧**。控制器知道证据过期，却仍选了"拿已有帧重解"那一族的动作 ——
那些帧描述的是**另一个窗口几何**，解出来必然维度不符。

与「被求解器跳掉」「预热预算耗尽」是同一个形状的第三例：
**一个在构造上不可能成功的动作被发了出去。**
"""
import json
import os
import pathlib
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abfe_preoptimizer import Stage2RepairController  # noqa: E402

from test_stage2_repair_controller import _mkrun, R4  # noqa: E402


def _run_with_stale_window(tmp_path, idx=1):
    """win{idx} 的产物是旧布局的（λ 个数与当前 window_ranges 对不上）。"""
    lam = [round(1.0 - 0.07 * i, 8) for i in range(13)]
    # 每个窗口的产物 λ 按当前布局切片写，路径版本记录也带上真实 λ ——
    # 只有这样 `_layout_matches` 才比得出"这份证据是不是当前布局的"。
    w = {i: {"K": R4[i][1] - R4[i][0], "lambdas_vdw": lam[R4[i][0]:R4[i][1]]}
         for i in range(4)}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13)
    pv = pathlib.Path(run) / "checkpoints" / "path_versions" / "v1.json"
    d = json.loads(pv.read_text())
    d["states"] = [{"id": f"s{i}", "lambda_vdw": lam[i]} for i in range(13)]
    d["lambdas_vdw"] = lam
    pv.write_text(json.dumps(d))

    # win{idx} 的产物少一个态 —— 它描述的是**另一套布局**（插 λ 之前的那套）
    f = (pathlib.Path(run) / "vanishing"
         / f"dual_window_{idx}_vdw_convergence.json")
    c = json.loads(f.read_text())
    c["lambdas_vdw"] = c["lambdas_vdw"][:-1]
    f.write_text(json.dumps(c))
    return run


def test_a_stale_layout_window_is_unresolved_not_eligible(tmp_path):
    run = _run_with_stale_window(tmp_path, idx=1)
    view = Stage2RepairController.for_physical_stage(run, "vanishing", "vdw").read()
    assert 1 in (view.get("stale_layout_evidence") or {}), (
        "布局变过的证据没被标成过期"
    )


def test_it_gets_resampling_never_a_re_solve_action(tmp_path):
    run = _run_with_stale_window(tmp_path, idx=1)
    plan = Stage2RepairController.for_physical_stage(
        run, "vanishing", "vdw").decide()

    assert plan["action"] == "RUN_PRODUCTION", plan["reason"]
    assert plan["action"] not in (
        "RECALIBRATE_FK", "PROBE_REANCHOR_EPOCH", "RELEARN_FK_EPOCH",
        "PROBE_CANDIDATE_FK",
    )
    assert plan["windows"] == [1]
    assert "维度不符" in plan["reason"] or "另一套 λ 布局" in plan["reason"]
    assert not plan["terminal"], "这是路由，不是终态"

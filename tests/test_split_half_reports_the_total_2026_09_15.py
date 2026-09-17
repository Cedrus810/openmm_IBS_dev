"""split-half 必须把**总漂移**和逐窗漂移一起报出来。

`split_half_drift_diagnostics` 一直同时算 `total_drift_kJ_mol`，但先前只有
逐窗那条进日志、进顶层键。缺了总量就无法区分两件事：

  · **真漂移**   —— 前后半程在算不同的数，总量也跟着动；
  · **归属重排** —— 总量稳，只是逐窗 segment ΔG 的拆分变了。

后者在本仓是**预期行为**：整条路径是一个拼接的全局 MBAR 解，窗口不是独立可分的
单元。拿逐窗漂移当"σ 低估"的证据（`sigma_inflated_from_split_half` 吃的正是它）
在这种情形下会系统性高估。

真机 cyclod_ligand2/rep1：逐窗报 +9.339 kJ/mol ⟹ σ 0.830→4.669（×5.63），
而总量那条**从来没被打印或落盘过**。
"""
import inspect

import ibs_engine as ie
import pytest

pytestmark = pytest.mark.cpu_only


def test_the_total_drift_is_surfaced_as_a_top_level_key():
    src = inspect.getsource(ie.solve_stage_integrated_from_dir) \
        if hasattr(ie, "solve_stage_integrated_from_dir") else None
    # 顶层键由写 res 的那个函数设置；按字符串在模块源码里找，避免绑死函数名
    mod = inspect.getsource(ie)
    assert 'res["split_half_total_z"]' in mod
    assert 'res["split_half_total_drift_kJ_mol"]' in mod
    # 逐窗那两个键**没有被删掉**（降级的是解读，不是数据）
    assert 'res["split_half_max_window_z"]' in mod
    assert 'res["split_half_diagnostics"] = drift' in mod


def test_the_warning_prints_the_total_next_to_the_per_window_one():
    mod = inspect.getsource(ie)
    assert "[split-half 总量]" in mod
    assert "total_delta_G_first_half_kJ_mol" in mod
    # 触发判据**不得**被顺手改掉 —— 仍是逐窗 max_z
    assert "if max_z is not None and max_z > SPLIT_HALF_DEFAULT_MAX_Z:" in mod


def test_the_sigma_inflation_says_which_quantity_it_eats():
    mod = inspect.getsource(ie)
    assert "本下界取自**逐窗**漂移" in mod


def test_the_diagnostics_function_still_computes_both():
    """降级的是**解读**，两个量都必须照算照返回。"""
    src = inspect.getsource(ie.split_half_drift_diagnostics)
    for k in ("total_drift_kJ_mol", "total_drift_over_2sigma",
              "max_window_drift_over_2sigma", "per_window"):
        assert f'"{k}"' in src, k
    # 分母是 2σ（两个半程各 SE≈√2σ，其差 SE≈2σ）—— 别顺手改成 σ
    assert "abs(second_half - first_half) / (2 * sigma)" in src

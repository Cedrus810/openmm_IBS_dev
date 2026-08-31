"""两条腿都不允许回退旧 IBS 主值（2026-08-30 决定）。

IBS 每个窗口只跑一条混合偏置轨迹，再把同一批帧计算成窗口内所有 λ 态的能量；
喂给 MBAR 时**所有目标物理态的 n_k 都是 0**。λ→0 端「水塌进配体空腔」的构型
从来没出现在输入帧里，任何权重计算都造不出它。

实测（4W53 溶剂腿，唯一变量是 Group5）：
    IBS 带 Group5    +35.606 ± 0.840
    IBS 去 Group5    +37.601 ± 1.278   <- Group5 不是主因
    逐态独立采样      -3.586 ± 0.221   （与 Beutler 参考差 0.10σ）

因此复合物腿当前的诚实状态是「没有可信 Stage2 数值」，而**不是**在 35.4 和 59.4
之间挑更接近实验的那个——按接近实验值挑数是拟合答案，不是计算它。
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_vanishing_forces_the_independent_endpoint_route():
    """独立端点段对 vanishing 必须是默认且强制，不是可选开关。"""
    src = (REPO / "abfe_pipeline.py").read_text(encoding="utf-8")
    i = src.index("_independent_endpoint_enabled = bool(")
    block = src[i:i + 500]
    assert 'kwargs.get("stage2_independent_endpoint_states"' not in block, (
        "不得再挂在可选 kwarg 上"
    )
    assert 'stage_name == "vanishing"' in block


def test_the_retired_switch_is_rejected_loudly():
    """还传旧开关的调用方必须收到明确报错，而不是被静默忽略。"""
    src = (REPO / "abfe_pipeline.py").read_text(encoding="utf-8")
    assert 'if "stage2_independent_endpoint_states" in kwargs:' in src
    i = src.index('if "stage2_independent_endpoint_states" in kwargs:')
    assert "已废弃" in src[i:i + 400]


def test_vanishing_without_enough_windows_fails_closed():
    """窗口数不足以拆出端点段时必须 fail-closed，不得退回纯 IBS。"""
    src = (REPO / "abfe_pipeline.py").read_text(encoding="utf-8")
    assert "vanishing 阶段必须启用独立端点段" in src


def test_bridge_rescue_cannot_overwrite_the_stitched_result():
    """rescue 会用完整 IBS 窗口集重解并覆盖 stage2——已用独立端点段时必须拒绝。

    🔑 [2026-08-31] 这条契约原来钉的是"判据出现在
    `combined_outputs = original_outputs + rescue_outputs` **之后** 2000 字符内"，
    也就是把 fail-closed 判据钉死在**整轮 rescue 采样跑完之后**——明知必被拒绝，
    还是先烧掉一轮 rescue ensembles 的 GPU 时间才抛错（与 [ATT-27] 同类缺陷）。
    判据已前移到进入 bridge rescue 分支的第一件事，所以本契约反过来要求：
    判据必须出现在 `_run_dual_lambda_stage("vanishing_rescue"` **之前**。
    """
    src = (REPO / "abfe_pipeline.py").read_text(encoding="utf-8")

    gate = src.index('"independent_endpoint_diagnostics"\n                    ) is not None')
    sampling = src.index('self._run_dual_lambda_stage(\n                            "vanishing_rescue"')
    assert gate < sampling, (
        "bridge rescue 的 fail-closed 判据必须在 "
        "_run_dual_lambda_stage(\"vanishing_rescue\") 之前求值，"
        "否则明知会拒绝还要先烧掉一整轮 rescue 采样"
    )
    assert "静默退回旧的纯 IBS 架构" in src[gate - 1200:gate + 1200]


def test_runabfe_fallback_refuses_ibs_only_vanishing():
    """回退分析路径不得成为绕过路由层的后门。"""
    import inspect
    import runabfe
    src = inspect.getsource(runabfe._analyze_dual_leg_artifacts)
    assert "_endpoint_analysis_artifact(" in src
    assert "回退分析拒绝用纯 IBS 窗口" in src
    assert "combine_ibs_and_independent_endpoint(" in src


def test_target_state_nk_is_zero_is_documented_where_it_happens():
    """求解器里那句『其余物理态样本数均为 0』是这个失效模式的直接证据，
    必须留在代码里，不能被后人当成无关注释删掉。"""
    src = (REPO / "ibs_engine.py").read_text(encoding="utf-8")
    assert "n_k_local[sampled_row] = n_frames" in src
    assert "其余物理态" in src and "样本数均为 0" in src

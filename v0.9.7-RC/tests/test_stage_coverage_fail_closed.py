"""P0-8 回归：预期窗口没有全部落盘时，必须拒绝求解截断的自由能曲线。

背景（改这块之前必读）：`GlobalMBARAnalyzer.solve_stage_integrated()` 的完整性
判据是 `len(local_results) == len(valid_windows)`，而 `valid_windows` 本身就是从
传进来的 `window_data` 算出来的——loader 静默丢掉的窗口在求解器里根本不可见。

协方差链**能**挡住中间缺窗：非首窗必须与已覆盖态共享一个 λ，否则走
`_fallback("window_overlap_broken_for_covariance_chain")`。挡不住的是两头：

  * 缺首窗 → 原来的 window 1 自然变成 `local_idx == 0`，走 `join_lam =
    local_lams[0]` 分支，链正常闭合；
  * 缺末窗 → 链提前正常结束。

两者都返回一条**截断**的 ΔG 并报 `converged=True`。`_assert_stage_result_sane()`
只检查求解器返回的字段（total_delta_G/total_error 有限、converged、min_overlap、
去相关样本数、端点不确定度），不数窗口也不查 λ 覆盖，结构上同样抓不到。

所以这道门必须加在 **loader 出口**——等数据到了求解器手里，信息已经丢了。
"""

import json

import numpy as np
import pytest

import ibs_engine as ie
from ibs_engine import (
    IBSIncompleteStageCoverageError,
    _assert_expected_windows_all_loaded,
)

pytestmark = pytest.mark.cpu_only


# ---------------------------------------------------------------------------
# 纯函数层
# ---------------------------------------------------------------------------


def _call(expected, loaded, missing=None):
    _assert_expected_windows_all_loaded(
        expected_windows=expected,
        loaded_windows=loaded,
        missing_windows=missing or [],
        source="unit-test",
    )


def test_complete_coverage_passes():
    _call([0, 1, 2, 3], [0, 1, 2, 3])


def test_loaded_order_does_not_matter():
    _call([0, 1, 2], [2, 0, 1])


@pytest.mark.parametrize(
    "missing_idx, why",
    [
        (0, "缺首窗：window 1 会变成协方差链的 local_idx==0，链照样闭合"),
        (3, "缺末窗：链提前结束，同样闭合"),
        (1, "缺中间窗：协方差链本来就能挡，但 loader 也不该放过"),
    ],
)
def test_any_missing_expected_window_fails_closed(missing_idx, why):
    loaded = [i for i in range(4) if i != missing_idx]
    with pytest.raises(IBSIncompleteStageCoverageError) as excinfo:
        _call(list(range(4)), loaded)
    message = str(excinfo.value)
    assert f"缺失窗口 [{missing_idx}]" in message, why
    # 报错必须同时给出 expected 与 loaded，否则没法判断到底截断了哪一段。
    assert "预期窗口 [0, 1, 2, 3]" in message
    assert f"实际加载 {loaded}" in message


def test_missing_file_detail_is_reported():
    with pytest.raises(IBSIncompleteStageCoverageError) as excinfo:
        _call(
            [0, 1],
            [1],
            missing=[{
                "window_index": 0,
                "lambda_range": [0, 5],
                "missing_files": ["bias", "base"],
            }],
        )
    message = str(excinfo.value)
    assert "λ范围=[0, 5]" in message
    assert "['bias', 'base']" in message


def test_explicitly_excluded_window_is_not_expected():
    """合法的部分分析（rescue ensemble 取代原始窗口）不能被这道门误伤。"""
    _call([0, 1, 3], [0, 1, 3])  # 窗口 2 已被调用方从 expected 里剔除


def test_zero_expected_windows_is_not_an_error_here():
    """空 expected 由调用侧的 `if not window_outputs` 负责，不在这道门里重复判。"""
    _call([], [])


# ---------------------------------------------------------------------------
# loader 层（真实落盘）
# ---------------------------------------------------------------------------

N_STATES_PER_WINDOW = 3
N_FRAMES = 40
RANGES = [(0, 3), (2, 5), (4, 7)]
LAMBDAS_VDW = [1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0]
LAMBDAS_COUL = [0.0] * len(LAMBDAS_VDW)


def _write_window(output_dir, checkpoint_dir, local_idx, stage_type="vdw", seed=0):
    """落一个完整窗口：energies/bias/base 三文件 + convergence manifest + ibs_state。"""
    rng = np.random.default_rng(20260727 + seed)
    u_kn = rng.normal(size=(N_STATES_PER_WINDOW, N_FRAMES))
    bias = rng.normal(size=N_FRAMES)
    base = rng.normal(size=N_FRAMES)

    paths = {}
    for label, array in (("energies", u_kn), ("bias", bias), ("base", base)):
        path = output_dir / f"dual_window_{local_idx}_{stage_type}_{label}.npy"
        np.save(path, array)
        paths[label] = path

    metadata = ie._window_data_metadata(
        str(paths["energies"]), str(paths["bias"]), str(paths["base"])
    )
    convergence_path = (
        output_dir / f"dual_window_{local_idx}_{stage_type}_convergence.json"
    )
    convergence_path.write_text(
        json.dumps({
            "window_data_protocol_version": ie.IBS_WINDOW_DATA_PROTOCOL_VERSION,
            "window_data": metadata,
        }),
        encoding="utf-8",
    )

    state_path = checkpoint_dir / f"ibs_state_{stage_type}_window_{local_idx}.json"
    state_path.write_text(
        json.dumps({"f_k": [0.0] * N_STATES_PER_WINDOW}), encoding="utf-8"
    )
    return paths


@pytest.fixture
def stage_dirs(tmp_path):
    output_dir = tmp_path / "vanishing"
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir.mkdir()
    checkpoint_dir.mkdir()
    for local_idx in range(len(RANGES)):
        _write_window(output_dir, checkpoint_dir, local_idx, seed=local_idx)
    return output_dir, checkpoint_dir


def _load(output_dir, checkpoint_dir, **kwargs):
    from abfe_pipeline import ABFEPipeline

    return ABFEPipeline._load_ibs_window_outputs_from_dir(
        str(output_dir),
        RANGES,
        LAMBDAS_COUL,
        LAMBDAS_VDW,
        checkpoint_dir=str(checkpoint_dir),
        **kwargs,
    )


def test_pipeline_loader_accepts_complete_stage(stage_dirs):
    outputs = _load(*stage_dirs)
    assert [o["window_index"] for o in outputs] == [0, 1, 2]
    assert all(o["f_k"].size == N_STATES_PER_WINDOW for o in outputs)


@pytest.mark.parametrize("missing_idx", [0, 1, 2])
def test_pipeline_loader_fails_closed_on_missing_window(stage_dirs, missing_idx):
    output_dir, checkpoint_dir = stage_dirs
    for label in ("energies", "bias", "base"):
        (output_dir / f"dual_window_{missing_idx}_vdw_{label}.npy").unlink()
    with pytest.raises(IBSIncompleteStageCoverageError, match=f"缺失窗口 \\[{missing_idx}\\]"):
        _load(output_dir, checkpoint_dir)


def test_pipeline_loader_fails_closed_when_only_bias_is_missing(stage_dirs):
    """三文件是不可分割的整体；只丢 bias 也不能降级成"这个窗口不存在"。"""
    output_dir, checkpoint_dir = stage_dirs
    (output_dir / "dual_window_0_vdw_bias.npy").unlink()
    with pytest.raises(IBSIncompleteStageCoverageError) as excinfo:
        _load(output_dir, checkpoint_dir)
    assert "['bias']" in str(excinfo.value)


def test_pipeline_loader_allows_explicitly_excluded_window(stage_dirs):
    """vanishing rescue 的合法路径：原始窗口被显式排除，由 rescue ensemble 补上。"""
    output_dir, checkpoint_dir = stage_dirs
    for label in ("energies", "bias", "base"):
        (output_dir / f"dual_window_1_vdw_{label}.npy").unlink()
    outputs = _load(output_dir, checkpoint_dir, excluded_local_windows={1})
    assert [o["window_index"] for o in outputs] == [0, 2]


def test_pipeline_loader_window_index_offset_does_not_break_the_gate(stage_dirs):
    """rescue 侧带 window_index_offset=10000；覆盖判定必须在 offset 之前做。"""
    output_dir, checkpoint_dir = stage_dirs
    outputs = _load(output_dir, checkpoint_dir, window_index_offset=10_000)
    assert [o["window_index"] for o in outputs] == [10_000, 10_001, 10_002]

    for label in ("energies", "bias", "base"):
        (output_dir / f"dual_window_2_vdw_{label}.npy").unlink()
    with pytest.raises(IBSIncompleteStageCoverageError, match="缺失窗口 \\[2\\]"):
        _load(output_dir, checkpoint_dir, window_index_offset=10_000)


def test_rescue_merge_positive_case(tmp_path):
    """original 排除失败窗口 + rescue 目录补上 —— 整条合并路径必须仍然通过。

    这是 P0-8 最容易误伤的一条：如果 expected 直接取 `self.ranges` 而不是显式
    参数，正常的 rescue 合并会被判成"缺窗"。
    """
    from abfe_pipeline import ABFEPipeline

    original_dir = tmp_path / "vanishing"
    original_ckpt = tmp_path / "checkpoints"
    rescue_dir = tmp_path / "vanishing_rescue"
    rescue_ckpt = tmp_path / "rescue_checkpoints"
    for d in (original_dir, original_ckpt, rescue_dir, rescue_ckpt):
        d.mkdir()

    failing = {1}
    for local_idx in range(len(RANGES)):
        if local_idx in failing:
            continue
        _write_window(original_dir, original_ckpt, local_idx, seed=local_idx)

    rescue_ranges = [(2, 5)]
    _write_window(rescue_dir, rescue_ckpt, 0, seed=99)

    original_outputs = ABFEPipeline._load_ibs_window_outputs_from_dir(
        str(original_dir), RANGES, LAMBDAS_COUL, LAMBDAS_VDW,
        checkpoint_dir=str(original_ckpt),
        excluded_local_windows=failing,
        window_label_prefix="original_window",
    )
    rescue_outputs = ABFEPipeline._load_ibs_window_outputs_from_dir(
        str(rescue_dir), rescue_ranges, LAMBDAS_COUL, LAMBDAS_VDW,
        checkpoint_dir=str(rescue_ckpt),
        window_index_offset=10_000,
        window_label_prefix="rescue_window",
    )
    combined = original_outputs + rescue_outputs
    assert [o["window_index"] for o in combined] == [0, 2, 10_000]
    covered = sorted({i for o in combined for i in o["lambda_indices"]})
    assert covered == list(range(7)), "合并后必须覆盖完整 λ 区间"

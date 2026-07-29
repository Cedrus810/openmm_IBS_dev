"""窗口产物复用 / resume 门 / Shadow-Bridge s 参数的离线契约测试。

三个被测对象（都不需要 GPU，也不建任何真实 OpenMM Context）：

  5. `abfe_pipeline.ABFEPipeline._invalidate_stage_window_files` 的 reuse_map
     —— λ 路径变了以后，哪些窗口产物可以按"λ 集合相同"改名复用、哪些必须清掉。
  6. `ibs_engine._resume_cached_window_gate_status` 的 8 个门
     —— 一份磁盘缓存在 resume 时能不能复用。这个纯函数是从 `run_all_windows`
     里那 ~110 行内联判断抽出来的（逐门语义未改），抽出来的唯一目的就是让这些
     门可以用 mock 的 convergence.json 单独验证 fail-closed。
  8. `ibs_engine.ShadowBridgeREMDManager._set_context_state` / `_context_lambda_label`
     —— 交换循环唯一接触 λ 的两个方法，必须只驱动 lambda_bridge_s。

`_run_stage_with_overlap_autorepair` 的 non_mutating_v1 短路**不在这里**：
`test_non_mutating_policy.py::test_non_mutating_runs_once_and_returns` 已经断言
run_once 恰好调用一次、结果原样返回、且所有 mutator 一碰即 AssertionError。
"""

import json
import os
import types
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

pytest.importorskip("openmm")

import abfe_pipeline as ap
import ibs_engine as ie


# ============================================================================
# 5. _invalidate_stage_window_files 的 reuse_map
# ============================================================================

STAGE_NAME = "vanishing"
STAGE_TYPE = "vdw"


@pytest.mark.parametrize("resume", [False, True])
def test_traditional_run_full_passes_resume_to_both_legs(tmp_path, resume):
    pipeline = ap.TraditionalABFEPipeline.__new__(ap.TraditionalABFEPipeline)
    pipeline.output_dir = str(tmp_path)
    calls = []

    def fake_run_leg(
        self,
        stage_name,
        lambdas_coul,
        lambdas_vdw,
        n_steps,
        *,
        resume,
        boresch_params,
        potential_type,
    ):
        calls.append((stage_name, resume))
        return {"delta_G": 1.0, "error": 0.1}

    pipeline.run_leg = types.MethodType(fake_run_leg, pipeline)
    pipeline.run_full(n_lambda=3, n_steps_per_leg=10, resume=resume)
    assert calls == [("decharging", resume), ("vanishing", resume)]


def test_traditional_cli_disables_resume_under_reset():
    source = (Path(__file__).resolve().parents[1] / "runabfe.py").read_text(
        encoding="utf-8"
    )
    start = source.index("def run_traditional_mode(")
    end = source.index("\ndef _load_frozen_stage_result(", start)
    traditional_body = source[start:end]
    assert traditional_body.count("resume=config.resume and not config.reset,") == 2


def _make_fake_pipeline(tmp_path):
    """只挂上 _invalidate_stage_window_files 真正会碰的东西。

    沿用 `test_non_mutating_policy.py::_make_fake_pipeline` 的 SimpleNamespace +
    types.MethodType 模式，不另造一套 fake 体系。`_window_lambda_key` 是
    @staticmethod，从类上取到的就是普通函数，直接挂到实例上即可按
    `self._window_lambda_key(...)` 调用。
    """
    fake = types.SimpleNamespace()
    fake.output_dir = str(tmp_path / "out")
    fake.checkpoint_dir = str(tmp_path / "ckpt")
    fake.logged = []
    fake._log = lambda msg, *a, **k: fake.logged.append(str(msg))
    fake._window_lambda_key = ap.ABFEPipeline._window_lambda_key
    fake._invalidate_stage_window_files = types.MethodType(
        ap.ABFEPipeline._invalidate_stage_window_files, fake
    )
    os.makedirs(os.path.join(fake.output_dir, STAGE_NAME), exist_ok=True)
    os.makedirs(fake.checkpoint_dir, exist_ok=True)
    return fake


def _current_accounting_conv(window_idx, **overrides):
    """一份"记账口径全部与当前一致"的 convergence.json 内容。

    `_old_window_accounting_ok` 要求 wca / ibs_bias / lj_tail_lrc 三个协议版本
    **加上** sampling_repair_policy 四项全部匹配，任何一项不符（或读不到文件）
    都保守判"不可复用"。版本号从模块常量读，不硬编码——否则递增版本时这个测试
    会变成"锁死旧版本"的假绿。
    """
    conv = {
        "window_idx": int(window_idx),
        "wca_accounting_version": ap.WCA_ACCOUNTING_VERSION,
        "ibs_bias_protocol_version": ap.IBS_BIAS_PROTOCOL_VERSION,
        "lj_tail_lrc_protocol_version": ap.TRADITIONAL_LJ_LRC_PROTOCOL_VERSION,
        "sampling_repair_policy": "non_mutating_v1",
    }
    conv.update(overrides)
    return conv


def _window_paths(fake, idx):
    stage_dir = os.path.join(fake.output_dir, STAGE_NAME)
    return {
        "energies": os.path.join(stage_dir, f"dual_window_{idx}_{STAGE_TYPE}_energies.npy"),
        "bias": os.path.join(stage_dir, f"dual_window_{idx}_{STAGE_TYPE}_bias.npy"),
        "base": os.path.join(stage_dir, f"dual_window_{idx}_{STAGE_TYPE}_base.npy"),
        "convergence": os.path.join(
            stage_dir, f"dual_window_{idx}_{STAGE_TYPE}_convergence.json"
        ),
        "ibs_state": os.path.join(
            fake.checkpoint_dir, f"ibs_state_{STAGE_TYPE}_window_{idx}.json"
        ),
    }


def _write_window(fake, idx, marker, conv=None):
    """给窗口 idx 写全 5 类产物。marker 写进 .npy，用来验证"哪份数据落到了哪个 idx"。"""
    paths = _window_paths(fake, idx)
    np.save(paths["energies"], np.full((3, 4), float(marker)))
    np.save(paths["bias"], np.full(4, float(marker)))
    np.save(paths["base"], np.full(4, float(marker)))
    with open(paths["convergence"], "w", encoding="utf-8") as fh:
        json.dump(_current_accounting_conv(idx) if conv is None else conv, fh)
    with open(paths["ibs_state"], "w", encoding="utf-8") as fh:
        json.dump({"marker": marker}, fh)
    return paths


def _marker_at(fake, idx):
    paths = _window_paths(fake, idx)
    if not os.path.exists(paths["energies"]):
        return None
    return float(np.load(paths["energies"])[0, 0])


def _all_exist(fake, idx):
    return all(os.path.exists(p) for p in _window_paths(fake, idx).values())


def _none_exist(fake, idx):
    return not any(os.path.exists(p) for p in _window_paths(fake, idx).values())


def test_reuse_map_renames_window_whose_lambda_set_is_unchanged(tmp_path):
    """λ 路径插入了一个新点，某个窗口覆盖的 λ 集合完全没变但 window_idx 挪了位置。

    这种窗口必须**改名复用**（不重新采样），且 convergence.json 里的 window_idx
    字段要跟着更新到新编号。
    """
    fake = _make_fake_pipeline(tmp_path)

    old_lambdas = [0.0, 0.25, 0.50, 1.00]
    old_ranges = [(0, 2), (2, 4)]          # 窗口0={0.0,0.25}  窗口1={0.50,1.00}
    # 在最前面插入一个新 λ=0.10：老窗口1 的 λ 集合 {0.50, 1.00} 原封不动，
    # 但在新方案里变成了窗口 2。
    new_lambdas = [0.0, 0.10, 0.25, 0.50, 1.00]
    new_ranges = [(0, 2), (1, 3), (3, 5)]  # 新窗口2={0.50,1.00} <- 老窗口1

    _write_window(fake, 0, marker=100.0)
    _write_window(fake, 1, marker=111.0)

    fake._invalidate_stage_window_files(
        STAGE_NAME, STAGE_TYPE,
        old_lambdas=old_lambdas, old_ranges=old_ranges,
        new_lambdas=new_lambdas, new_ranges=new_ranges,
    )

    assert _all_exist(fake, 2), "λ 集合未变的窗口应被改名复用到新 idx=2"
    assert _marker_at(fake, 2) == 111.0, "落到 idx=2 的应该是老窗口1 的数据"
    assert _none_exist(fake, 1), "老 idx=1 的产物应已被搬走（不是留在原地）"
    assert _none_exist(fake, 0), "λ 集合变了的窗口0 必须被清掉，等待重采"

    with open(_window_paths(fake, 2)["convergence"], encoding="utf-8") as fh:
        assert json.load(fh)["window_idx"] == 2, "convergence.json 的 window_idx 未更新"


def test_reuse_map_handles_index_swap_without_clobbering(tmp_path):
    """new1←old3 且 new3←old1 的**交换**场景：两份数据都要正确落位。

    代码用 `.reuse_tmp` 两阶段搬运正是为了这个——单阶段直接 os.replace 会让两个
    窗口互相覆盖，最后两个 idx 拿到同一份数据（且另一份永久丢失）。
    """
    fake = _make_fake_pipeline(tmp_path)

    # 四个宽度相同的窗口，构造一个让 λ 集合位置互换的新方案。
    old_lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    old_ranges = [(0, 2), (2, 4), (4, 6), (6, 8)]
    # 新 λ 路径把 old3 的 λ 集合 {0.6,0.7} 放到 new1，old1 的 {0.2,0.3} 放到 new3。
    new_lambdas = [0.0, 0.1, 0.6, 0.7, 0.4, 0.5, 0.2, 0.3]
    new_ranges = [(0, 2), (2, 4), (4, 6), (6, 8)]

    for idx in range(4):
        _write_window(fake, idx, marker=float(idx))

    fake._invalidate_stage_window_files(
        STAGE_NAME, STAGE_TYPE,
        old_lambdas=old_lambdas, old_ranges=old_ranges,
        new_lambdas=new_lambdas, new_ranges=new_ranges,
    )

    assert _marker_at(fake, 1) == 3.0, "new1 应拿到 old3 的数据"
    assert _marker_at(fake, 3) == 1.0, "new3 应拿到 old1 的数据"
    # 未参与交换的两个窗口 λ 集合原地未变，应留在原 idx。
    assert _marker_at(fake, 0) == 0.0
    assert _marker_at(fake, 2) == 2.0
    for idx in range(4):
        assert _all_exist(fake, idx), f"窗口 {idx} 的 5 类产物应齐全"


@pytest.mark.parametrize(
    "conv_override, reason",
    [
        ({"wca_accounting_version": 999}, "base/bias 力组切分口径变了"),
        ({"ibs_bias_protocol_version": -1}, "IBS 偏置预热/冻结协议变了"),
        ({"lj_tail_lrc_protocol_version": 999}, "LJ 长程尾项公式变了"),
        ({"sampling_repair_policy": "legacy_mutating"}, "旧变异策略的 f_k 参考系不同"),
        ({"sampling_repair_policy": None}, "旧缓存没有这个字段"),
    ],
)
def test_reuse_requires_every_accounting_version_to_match(tmp_path, conv_override, reason):
    """λ 集合再怎么完全相同，任一记账口径不符就必须删除、重采，不许复用。"""
    fake = _make_fake_pipeline(tmp_path)

    old_lambdas = [0.0, 0.25, 0.50, 1.00]
    old_ranges = [(0, 2), (2, 4)]
    new_lambdas = [0.0, 0.10, 0.25, 0.50, 1.00]
    new_ranges = [(0, 2), (1, 3), (3, 5)]

    conv = _current_accounting_conv(1)
    conv.update(conv_override)
    _write_window(fake, 0, marker=100.0)
    _write_window(fake, 1, marker=111.0, conv=conv)

    fake._invalidate_stage_window_files(
        STAGE_NAME, STAGE_TYPE,
        old_lambdas=old_lambdas, old_ranges=old_ranges,
        new_lambdas=new_lambdas, new_ranges=new_ranges,
    )

    assert _none_exist(fake, 1), f"{reason}：旧产物必须被清掉"
    assert _none_exist(fake, 2), f"{reason}：不得被改名复用到新 idx"


def test_reuse_treats_unreadable_convergence_as_mismatch(tmp_path):
    """convergence.json 损坏/缺失时 `_old_window_accounting_ok` 保守判不一致。"""
    fake = _make_fake_pipeline(tmp_path)

    old_lambdas = [0.0, 0.25, 0.50, 1.00]
    old_ranges = [(0, 2), (2, 4)]
    new_lambdas = [0.0, 0.10, 0.25, 0.50, 1.00]
    new_ranges = [(0, 2), (1, 3), (3, 5)]

    _write_window(fake, 0, marker=100.0)
    paths = _write_window(fake, 1, marker=111.0)
    with open(paths["convergence"], "w", encoding="utf-8") as fh:
        fh.write("{ this is not valid json")

    fake._invalidate_stage_window_files(
        STAGE_NAME, STAGE_TYPE,
        old_lambdas=old_lambdas, old_ranges=old_ranges,
        new_lambdas=new_lambdas, new_ranges=new_ranges,
    )
    assert _none_exist(fake, 1)
    assert _none_exist(fake, 2)


def test_missing_reuse_arguments_falls_back_to_wiping_everything(tmp_path):
    """不提供 old/new lambdas+ranges 时退化为原来的"全部清空"行为。"""
    fake = _make_fake_pipeline(tmp_path)
    for idx in range(3):
        _write_window(fake, idx, marker=float(idx))

    fake._invalidate_stage_window_files(STAGE_NAME, STAGE_TYPE)

    for idx in range(3):
        assert _none_exist(fake, idx), f"窗口 {idx} 应被全部清掉"


def test_unclaimed_stale_window_indices_are_removed(tmp_path):
    """新方案窗口数变少时，没被任何 new_idx 认领的旧编号残留必须清掉。

    否则磁盘上会留下"编号看起来合法、内容属于另一段 λ"的孤儿文件，之后极易
    被别处误读。
    """
    fake = _make_fake_pipeline(tmp_path)

    old_lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    old_ranges = [(0, 2), (2, 4), (4, 6)]
    # 新方案只有两个窗口，且第一个 λ 集合与老窗口0 完全一致。
    new_lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    new_ranges = [(0, 2), (2, 6)]

    for idx in range(3):
        _write_window(fake, idx, marker=float(idx))

    fake._invalidate_stage_window_files(
        STAGE_NAME, STAGE_TYPE,
        old_lambdas=old_lambdas, old_ranges=old_ranges,
        new_lambdas=new_lambdas, new_ranges=new_ranges,
    )

    assert _marker_at(fake, 0) == 0.0, "λ 集合未变的窗口0 应原地保留"
    assert _none_exist(fake, 1), "λ 集合变了的窗口1 应被清掉"
    assert _none_exist(fake, 2), "新方案里不存在的 idx=2 残留必须被清掉"


def test_reuse_logs_what_it_reused_and_removed(tmp_path):
    """复用/清理都必须留下日志——静默搬运文件是事后无法复盘的。"""
    fake = _make_fake_pipeline(tmp_path)

    old_lambdas = [0.0, 0.25, 0.50, 1.00]
    old_ranges = [(0, 2), (2, 4)]
    new_lambdas = [0.0, 0.10, 0.25, 0.50, 1.00]
    new_ranges = [(0, 2), (1, 3), (3, 5)]

    _write_window(fake, 0, marker=100.0)
    _write_window(fake, 1, marker=111.0)
    fake._invalidate_stage_window_files(
        STAGE_NAME, STAGE_TYPE,
        old_lambdas=old_lambdas, old_ranges=old_ranges,
        new_lambdas=new_lambdas, new_ranges=new_ranges,
    )

    joined = "\n".join(fake.logged)
    assert "复用" in joined
    assert "清理" in joined


# ============================================================================
# 6. _resume_cached_window_gate_status 的 8 个门
# ============================================================================

LC_WIN = [1.0, 0.9, 0.8]
LV_WIN = [1.0, 1.0, 1.0]
LSE_TOL = 0.05
TARGET_STEPS = 500_000
GOOD_SHAPE = (len(LC_WIN), 4000)
REPAIR_POLICY = "non_mutating_v1"

EARLY_STOP_CONFIG = {
    "min_steps": 100_000,
    "check_interval_steps": 25_000,
    "required_consecutive_passes": 2,
    "min_ess_ratio": 0.05,
    "min_absolute_ess": 50.0,
    "min_decorrelated_samples": 20,
    "max_delta_g_drift_kJ_mol": 0.5,
    "max_uncertainty_kJ_mol": 1.0,
}


def _matching_conv(**overrides):
    """一份八门全过的 convergence.json。协议版本一律从模块常量读。"""
    conv = {
        "lambdas_coul": list(LC_WIN),
        "lambdas_vdw": list(LV_WIN),
        "wca_accounting_version": ie.WCA_ACCOUNTING_VERSION,
        "ibs_bias_protocol_version": ie.IBS_BIAS_PROTOCOL_VERSION,
        "lse_log_residual_tolerance": LSE_TOL,
        "lj_tail_lrc_protocol_version": ie.TRADITIONAL_LJ_LRC_PROTOCOL_VERSION,
        "sampling_repair_policy": REPAIR_POLICY,
        "n_steps_per_window_effective": TARGET_STEPS,
        "early_stop_triggered": False,
    }
    conv.update(overrides)
    return conv


def _gate(conv, shape=GOOD_SHAPE, enable_early_stop=False, target_steps=TARGET_STEPS,
          early_stop_config=None, lse_tolerance=LSE_TOL):
    return ie._resume_cached_window_gate_status(
        conv,
        shape,
        LC_WIN,
        LV_WIN,
        REPAIR_POLICY,
        lse_tolerance,
        enable_early_stop,
        dict(EARLY_STOP_CONFIG if early_stop_config is None else early_stop_config),
        target_steps,
    )


def test_resume_accepts_fully_matching_cache():
    status = _gate(_matching_conv())
    assert status["usable"] is True, f"八门全过的缓存应可复用: {status}"
    assert status["reason"] is None
    for gate_name in (
        "shape_ok", "lambdas_match", "version_match", "bias_protocol_match",
        "lse_tolerance_match", "lrc_version_match", "repair_policy_match",
        "early_stop_ok",
    ):
        assert status[gate_name] is True, f"{gate_name} 应为 True"


@pytest.mark.parametrize(
    "overrides, failing_gate",
    [
        ({"lambdas_coul": [1.0, 0.9, 0.7]}, "lambdas_match"),
        ({"lambdas_vdw": [1.0, 1.0, 0.5]}, "lambdas_match"),
        ({"lambdas_coul": [1.0, 0.9]}, "lambdas_match"),
        ({"lambdas_vdw": [1.0, 1.0, 1.0, 1.0]}, "lambdas_match"),
        ({"wca_accounting_version": 999}, "version_match"),
        ({"ibs_bias_protocol_version": -7}, "bias_protocol_match"),
        ({"lse_log_residual_tolerance": LSE_TOL * 1.5}, "lse_tolerance_match"),
        ({"lj_tail_lrc_protocol_version": 999}, "lrc_version_match"),
        ({"sampling_repair_policy": "legacy_mutating"}, "repair_policy_match"),
    ],
)
def test_resume_rejects_each_gate_independently(overrides, failing_gate):
    """逐门打偏一次，验证只有该门变 False、整体不可复用、且给出非空原因。"""
    status = _gate(_matching_conv(**overrides))
    assert status["usable"] is False, f"{failing_gate} 打偏后仍判可复用: {status}"
    assert status[failing_gate] is False, f"{failing_gate} 应为 False: {status}"
    assert status["reason"], "拒绝复用必须给出非空原因"


@pytest.mark.parametrize(
    "missing_field, failing_gate",
    [
        ("lambdas_coul", "lambdas_match"),
        ("lambdas_vdw", "lambdas_match"),
        ("wca_accounting_version", "version_match"),
        ("ibs_bias_protocol_version", "bias_protocol_match"),
        ("lse_log_residual_tolerance", "lse_tolerance_match"),
        ("lj_tail_lrc_protocol_version", "lrc_version_match"),
        ("sampling_repair_policy", "repair_policy_match"),
        ("n_steps_per_window_effective", "early_stop_ok"),
    ],
)
def test_resume_fails_closed_on_missing_fields(missing_field, failing_gate):
    """**逐字段删除**（不是改值）：旧格式缓存缺这些 key 时必须保守重采。

    这条和上面那条不能互相替代——"字段值不对"和"字段根本不存在"走的是不同分支，
    历史上 fail-closed 漏掉的正是后者（`.get()` 返回 None 却被当成通过）。
    """
    conv = _matching_conv()
    conv.pop(missing_field)
    status = _gate(conv)
    assert status["usable"] is False, f"缺 {missing_field} 仍判可复用: {status}"
    assert status[failing_gate] is False
    assert status["reason"]


def test_resume_rejects_empty_convergence_json():
    """整份 conv 为空（或不是 dict）时必须全门失败，不许崩。"""
    for conv in ({}, None, "not-a-dict", []):
        status = _gate(conv)
        assert status["usable"] is False, f"conv={conv!r} 仍判可复用"
        assert status["reason"]


def test_resume_accepts_lse_tolerance_within_roundoff():
    """LSE 容差只接受 roundoff 级别的差异（atol=1e-15, rtol=0）。"""
    assert _gate(_matching_conv(lse_log_residual_tolerance=LSE_TOL + 1e-16))[
        "lse_tolerance_match"
    ] is True
    assert _gate(_matching_conv(lse_log_residual_tolerance=LSE_TOL + 1e-9))[
        "lse_tolerance_match"
    ] is False


def test_resume_rejects_cache_produced_under_lower_step_budget():
    """缓存产出时的目标步数低于当前目标 → 拒绝，**即使从未触发 early stop**。

    这条门不管 early_stop_triggered 是 True 还是 False 都要过：一份"250k 时代
    完整跑满"的窗口，不能被静默当成满足"预算已提到 500k"的当前要求。
    """
    status = _gate(
        _matching_conv(n_steps_per_window_effective=250_000, early_stop_triggered=False),
        target_steps=500_000,
    )
    assert status["usable"] is False
    assert status["early_stop_ok"] is False
    assert "目标步数" in status["early_stop_reject_reason"]


def test_resume_accepts_cache_produced_under_higher_step_budget():
    """缓存跑得比当前要求更多是可以接受的（门是"不得低于"，不是"必须相等"）。"""
    status = _gate(
        _matching_conv(n_steps_per_window_effective=900_000), target_steps=500_000
    )
    assert status["usable"] is True


def test_resume_rejects_early_stopped_cache_when_early_stop_now_disabled():
    conv = _matching_conv(
        early_stop_triggered=True,
        early_stop_protocol_version=ie.EARLY_STOP_PROTOCOL_VERSION,
        early_stop_config=dict(EARLY_STOP_CONFIG),
    )
    status = _gate(conv, enable_early_stop=False)
    assert status["usable"] is False
    assert status["early_stop_ok"] is False
    assert "未启用 early stop" in status["early_stop_reject_reason"]


def test_resume_rejects_early_stopped_cache_on_protocol_version_mismatch():
    conv = _matching_conv(
        early_stop_triggered=True,
        early_stop_protocol_version=ie.EARLY_STOP_PROTOCOL_VERSION + 100,
        early_stop_config=dict(EARLY_STOP_CONFIG),
    )
    status = _gate(conv, enable_early_stop=True)
    assert status["usable"] is False
    assert "early_stop_protocol_version" in status["early_stop_reject_reason"]


def test_resume_early_stop_cache_requires_matching_config_not_just_version():
    """协议版本一致还不够：阈值从松调紧时旧的短样本缓存也必须拒绝。

    max_uncertainty_kJ_mol 从缓存记录的 5.0 收紧到当前 1.0 —— 协议版本号根本不
    会变（判据逻辑没变，只是阈值变了），但旧窗口是在更松的阈值下通过 early stop
    的，不能假设它在新阈值下也一定通过。
    """
    loose_config = dict(EARLY_STOP_CONFIG)
    loose_config["max_uncertainty_kJ_mol"] = 5.0
    conv = _matching_conv(
        early_stop_triggered=True,
        early_stop_protocol_version=ie.EARLY_STOP_PROTOCOL_VERSION,
        early_stop_config=loose_config,
    )
    status = _gate(conv, enable_early_stop=True)   # 当前用的是 1.0
    assert status["usable"] is False
    assert status["early_stop_ok"] is False
    assert "early_stop_config" in status["early_stop_reject_reason"]


def test_resume_accepts_early_stopped_cache_with_identical_config():
    conv = _matching_conv(
        early_stop_triggered=True,
        early_stop_protocol_version=ie.EARLY_STOP_PROTOCOL_VERSION,
        early_stop_config=dict(EARLY_STOP_CONFIG),
    )
    status = _gate(conv, enable_early_stop=True)
    assert status["usable"] is True, f"配置完全一致的 early-stop 缓存应可复用: {status}"


def test_resume_rejects_early_stopped_cache_with_missing_config():
    """early_stop_triggered=True 但没记 early_stop_config → 无从核对 → 拒绝。"""
    conv = _matching_conv(
        early_stop_triggered=True,
        early_stop_protocol_version=ie.EARLY_STOP_PROTOCOL_VERSION,
    )
    status = _gate(conv, enable_early_stop=True)
    assert status["usable"] is False
    assert status["early_stop_ok"] is False


@pytest.mark.parametrize(
    "shape, reason",
    [
        ((len(LC_WIN) + 1, 4000), "物理态数与本次窗口 λ 数不符"),
        ((len(LC_WIN) - 1, 4000), "物理态数偏少"),
        ((len(LC_WIN), 0), "一帧都没有"),
        ((len(LC_WIN),), "ndim != 2"),
        ((len(LC_WIN), 10, 2), "ndim != 2"),
        (None, "拿不到 shape"),
    ],
)
def test_resume_rejects_wrong_energy_shape(shape, reason):
    status = _gate(_matching_conv(), shape=shape)
    assert status["usable"] is False, f"{reason} 仍判可复用"
    assert status["shape_ok"] is False


def test_resume_reason_names_the_gate_that_actually_failed():
    """reason 的判定顺序必须与调用侧那串 elif 诊断打印一致——"日志说的原因"
    和"函数报告的原因"永远是同一个门，否则排障会被误导。"""
    cases = [
        ({"wca_accounting_version": 999}, "wca_accounting_version"),
        ({"ibs_bias_protocol_version": -7}, "ibs_bias_protocol_version"),
        ({"lj_tail_lrc_protocol_version": 999}, "lj_tail_lrc_protocol_version"),
        ({"sampling_repair_policy": "legacy_mutating"}, "sampling_repair_policy"),
        ({"lambdas_coul": [1.0, 0.9, 0.7]}, "λ 值不匹配"),
    ]
    for overrides, expected_fragment in cases:
        status = _gate(_matching_conv(**overrides))
        assert expected_fragment in status["reason"], (
            f"{overrides} 的 reason={status['reason']!r} 未点名 {expected_fragment}"
        )


def test_resume_gate_does_not_mutate_the_input_dict():
    """纯函数不得改调用方传进来的 cached_conv（它随后还要被写回/记诊断）。"""
    conv = _matching_conv()
    snapshot = json.dumps(conv, sort_keys=True)
    _gate(conv)
    assert json.dumps(conv, sort_keys=True) == snapshot


# ============================================================================
# 8. ShadowBridgeREMDManager 的 s 参数
# ============================================================================

LAMBDAS_S = [0.0, 0.35, 0.7, 1.0]


class _RecordingContext:
    """记录 setParameter 调用的假 Context。

    `fail_with` 用来验证 `_try_set_context_parameter` 的异常边界：它只允许吞掉
    含 "invalid parameter name" 的错误，其它异常必须原样抛出。
    """

    def __init__(self, fail_with=None):
        self.calls = []
        self._fail_with = fail_with

    def setParameter(self, name, value):
        if self._fail_with is not None:
            raise self._fail_with
        self.calls.append((str(name), float(value)))


def _make_bridge_manager(
    lambdas_s=LAMBDAS_S,
    s_param_name="lambda_bridge_s",
    context_to_state=None,
    contexts=None,
):
    """绕过 __init__ 构造一个 ShadowBridgeREMDManager。

    真正的 __init__ 会建 n 个真实 OpenMM Context（还会 deepcopy 系统），而被测的
    两个方法只读 contexts / s_param_name / lambdas_bridge_s / _context_to_state
    四个字段。用 object.__new__ 只填这四个，测试因此完全不碰 GPU/系统构建，同时
    仍然是**真实类**的实例——`_try_set_context_parameter` 等继承来的方法照常生效。
    """
    manager = object.__new__(ie.ShadowBridgeREMDManager)
    manager.lambdas_bridge_s = np.asarray(lambdas_s, dtype=float)
    manager.s_param_name = s_param_name
    manager.contexts = (
        list(contexts) if contexts is not None
        else [_RecordingContext() for _ in lambdas_s]
    )
    manager._context_to_state = list(
        range(len(lambdas_s)) if context_to_state is None else context_to_state
    )
    return manager


def test_set_context_state_takes_lambda_from_state_and_writes_to_context():
    """用 state_idx 取 λ、写进 context_idx 指定的那个 Context。

    REMD 交换之后 context_idx 与 state_idx 就不再相等了；把这两个索引搞混会让
    整条 bridge 腿在错误的 s 值下采样，而且不会报任何错——只有事后 ΔG 对不上。
    """
    manager = _make_bridge_manager()
    manager._set_context_state(2, 0)

    assert manager.contexts[2].calls == [("lambda_bridge_s", LAMBDAS_S[0])], (
        f"应把 state 0 的 s 值写进 context 2: {manager.contexts[2].calls}"
    )
    for idx in (0, 1, 3):
        assert manager.contexts[idx].calls == [], f"context {idx} 不该被写"


def test_set_context_state_writes_only_the_bridge_parameter():
    """绝不能写 lambda_coul / lambda_vdw。

    基类 REMDManager._set_context_state 写的就是这两个参数，而基类 __init__ 是用
    占位的 lambdas_coul=[0]*n / lambdas_vdw=[1]*n 调的——一旦子类没有完整覆盖这个
    方法（或某次重构把 super() 调用加回来），那些占位值就会被写进 Context，静默
    破坏整个 bridge 腿的哈密顿量。
    """
    manager = _make_bridge_manager()
    for state_idx in range(len(LAMBDAS_S)):
        manager._set_context_state(state_idx, state_idx)

    written = {name for ctx in manager.contexts for name, _ in ctx.calls}
    assert written == {"lambda_bridge_s"}, f"只应写 lambda_bridge_s，实际写了 {written}"


def test_set_context_state_honours_custom_s_param_name():
    manager = _make_bridge_manager(s_param_name="lambda_custom_s")
    manager._set_context_state(1, 3)
    assert manager.contexts[1].calls == [("lambda_custom_s", LAMBDAS_S[3])]


def test_set_context_state_covers_every_state_value():
    """逐个 state 写一遍，确认取的是 lambdas_bridge_s[state_idx] 而不是别的索引。"""
    manager = _make_bridge_manager()
    for state_idx in range(len(LAMBDAS_S)):
        manager._set_context_state(0, state_idx)
    assert [value for _, value in manager.contexts[0].calls] == LAMBDAS_S


def test_set_context_state_swallows_only_invalid_parameter_name():
    """`_try_set_context_parameter` 只吞 "invalid parameter name"。

    其它异常（例如 CUDA 报错）必须向上传播——静默吃掉会让一个坏掉的 Context
    继续参与交换循环，产出看起来正常的垃圾数据。
    """
    tolerated = _make_bridge_manager(
        contexts=[_RecordingContext(fail_with=Exception(
            "Context.setParameter(): invalid parameter name lambda_bridge_s"
        ))] + [_RecordingContext() for _ in LAMBDAS_S[1:]]
    )
    tolerated._set_context_state(0, 1)      # 不应抛

    fatal = _make_bridge_manager(
        contexts=[_RecordingContext(fail_with=RuntimeError("CUDA error: launch failed"))]
        + [_RecordingContext() for _ in LAMBDAS_S[1:]]
    )
    with pytest.raises(RuntimeError, match="CUDA error"):
        fatal._set_context_state(0, 1)


def test_context_lambda_label_resolves_state_through_the_swap_map():
    """标签必须经 _context_to_state 反查，且后两个返回值都是 s 值（不是 coul/vdw）。"""
    manager = _make_bridge_manager(context_to_state=[2, 0, 1, 3])

    state_idx, val_a, val_b = manager._context_lambda_label(0)
    assert state_idx == 2
    assert val_a == pytest.approx(LAMBDAS_S[2])
    assert val_b == pytest.approx(LAMBDAS_S[2]), "两个 λ 位都应报同一个 s 值"

    assert manager._context_lambda_label(1)[0] == 0
    assert manager._context_lambda_label(2)[0] == 1
    assert manager._context_lambda_label(3)[0] == 3


def test_context_lambda_label_returns_plain_floats():
    """返回值要能直接进 JSON 诊断——np.float64 会让 json.dump 报 TypeError。"""
    manager = _make_bridge_manager()
    state_idx, val_a, val_b = manager._context_lambda_label(1)
    assert type(state_idx) is int
    assert type(val_a) is float and type(val_b) is float
    json.dumps({"state": state_idx, "s": val_a})   # 不应抛

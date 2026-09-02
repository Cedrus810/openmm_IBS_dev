"""0831issue.md P2 批次回归（2026-09-01）。

只覆盖**行为有变**的那些条目；纯注释/文档类的不在这里钉。
分组与 `docs/RELEASE_READINESS_2026-08-31.md` 的「第九轮审查 backlog」一致。
"""

from __future__ import annotations

import ast
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

pytest.importorskip("openmm")

REPO = Path(__file__).resolve().parents[1]


# ===========================================================================
# A 组：会算错数的
# ===========================================================================


def test_cdf_no_longer_double_counts_the_second_to_last_weight():
    """`xp[-1] = 1.0` 覆盖式写法必须消失（两处副本），末端仍严格为 1.0。

    旧写法：`xp = [0] + cumsum(w)[:-1]/sum(w)`，再把末元素覆盖成 1.0 —— 那个赋值
    同时覆盖了 c_{N-2}，最后一个区间宽度从 w[N-2] 变成 w[N-2]+w[N-1]。
    """
    src = (REPO / "abfe_preoptimizer.py").read_text(encoding="utf-8")
    assert "xp[-1] = 1.0" not in src
    assert src.count("interval_weights = np.asarray(density_weight") == 2

    w = np.array([0.05, 0.10, 0.20, 0.40, 0.25])
    iw = w[:-1]
    xp = np.concatenate(([0.0], np.cumsum(iw) / float(np.sum(iw))))
    assert xp[0] == pytest.approx(0.0)
    assert xp[-1] == pytest.approx(1.0)
    # 最后一段是 0.40/0.75 ≈ 0.533，不是旧写法的 0.65。
    assert np.diff(xp)[-1] == pytest.approx(0.40 / 0.75)


def test_reduced_energies_indexes_lrc_by_physical_state_not_column_position():
    """LJ 尾项系数按物理 λ 态编址，不能用"记录列布局中的位置"去取。"""
    from ibs_engine import _reduced_energies_for_record

    kt = 2.494339
    n_frames = 3
    # 非恒等布局：列 0 是物理态 2，列 1 是物理态 0。
    record = {
        "energy_column_indices": [2, 0],
        "u_cv_kj_mol": np.zeros((n_frames, 2)),
        "volume_nm3": np.ones(n_frames),
    }
    lrc = np.array([100.0, 200.0, 300.0])   # 按物理态编址

    out = _reduced_energies_for_record(record, [2, 0], kt, lrc)
    # 物理态 2 → 300、物理态 0 → 100（除以 V=1，再 /kt）。
    np.testing.assert_allclose(out[0], np.array([300.0, 100.0]) / kt)

    # 越界必须 fail closed 而不是静默取错。
    with pytest.raises(IndexError):
        _reduced_energies_for_record(
            {**record, "energy_column_indices": [5, 0]},
            [5, 0], kt, lrc,
        )


def test_softcore_rejects_nonpositive_lambda_exponent():
    """n_lj <= 0 会让 λ=0 态的 LJ 尾项系数被静默置零。"""
    from abfe_core import ACESoftcorePotential
    from ibs_engine import _normalize_softcore_params

    ok = ACESoftcorePotential(alpha_lj=0.5, alpha_coul=0.5, power_lj=(2, 2), power_coul=(1, 1))
    norm = _normalize_softcore_params(ok, 41, explicit=True)
    assert norm.provenance["lambda_exponent_n_lj"] == 2

    bad = ACESoftcorePotential(alpha_lj=0.5, alpha_coul=0.5, power_lj=(2, 0), power_coul=(1, 1))
    with pytest.raises(ValueError, match="n_lj"):
        _normalize_softcore_params(bad, 41, explicit=True)


def test_explicit_softcore_exponents_are_not_silently_reset():
    """[0831issue P2 顺带发现] explicit 路径以前用 `getattr(..., "power_lj")` 重建，

    而 `ACESoftcorePotential` 把 power_lj 拆成 m_lj/n_lj 两个属性、根本没有
    `power_lj` 属性 —— getattr 永远落到默认 (2,2)，用户显式给的指数被静默重置。
    生产恰好就是 (2,2) 所以数值无变化，但这与本函数自己的 provenance 声明矛盾。
    """
    from abfe_core import ACESoftcorePotential
    from ibs_engine import _normalize_softcore_params

    src = ACESoftcorePotential(
        alpha_lj=0.5, alpha_coul=0.5, power_lj=(3, 4), power_coul=(2, 5)
    )
    norm = _normalize_softcore_params(src, 41, explicit=True)
    assert (norm.m_lj, norm.n_lj) == (3, 4), "λ_vdw 指数被静默重置"
    assert (norm.m_coul, norm.n_coul) == (2, 5), "λ_coul 指数被静默重置"
    assert norm.provenance["lambda_exponent_n_lj"] == 4


def test_unvalidated_variance_metric_entry_point_is_fail_closed():
    """`analyze_gradient_and_optimize_path` 按 PHY-08 同等处置禁用。

    它用 `Var(U_group1)` 而非生产 PME 的 `beta² Var[dU/dλ]`。唯一调用者
    `ABFEPipeline.run_preoptimization` 自身零调用方，生产不可达。
    """
    src = (REPO / "abfe_preoptimizer.py").read_text(encoding="utf-8")
    i = src.index("def analyze_gradient_and_optimize_path(")
    body = src[i:i + 3000]
    assert "raise RuntimeError(" in body
    assert "beta² Var[dU/dlambda]" in body or "beta² Var[dU/dλ]" in body
    # NaN 样本必须丢弃，不能替换成前值/0.0 后继续计入方差。
    assert "e = energies[-1] if energies else 0.0" not in src


def test_safe_boresch_ramp_uses_delta_energy_not_absolute():
    """`abs(总势能) > 1e5` 对 7 万原子盒恒真，会把正常体系全判失败。"""
    src = (REPO / "ibs_engine.py").read_text(encoding="utf-8")
    i = src.index("def _safe_boresch_ramp(")
    j = src.index("def get_stage_data_for_analysis(", i)
    fn = src[i:j]
    # 跳过 docstring（它会引用旧判据来解释为什么那是错的）。
    body = fn.split('"""', 2)[-1]
    assert "abs(e) > 1e5" not in body
    assert "delta_e = e - e_ref" in body
    assert "delta_e > max_delta_e" in body


# ===========================================================================
# B 组：口径 / 契约不一致
# ===========================================================================


def test_strip_unit_suffix_rejects_degrees():
    """`_deg` 后缀会被当弧度消费，释放项错约 57 倍 → 必须拒绝。"""
    src = (REPO / "abfe_pipeline.py").read_text(encoding="utf-8")
    i = src.index("def _strip_unit_suffix(")
    j = src.index("def apply_boresch_correction(", i)
    fn = src[i:j]
    assert 'suffixes = ["_kJ_mol_nm2", "_kJ_mol_rad2", "_nm", "_rad"]' in fn
    assert '"_deg"' in fn and "raise ValueError(" in fn

    # 直接执行该静态方法（纯字符串逻辑，不需要 OpenMM）。
    tree = ast.parse(src)
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ABFEPipeline"
    )
    node = next(
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_strip_unit_suffix"
    )
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"Optional": object, "Dict": dict}
    exec(compile(module, "<strip>", "exec"), ns)
    strip = ns["_strip_unit_suffix"]

    targets = {"r0_nm": "r0", "thetaA0_rad": "thetaA0"}
    assert strip("r0_nm", targets) == "r0"
    assert strip("thetaA0_rad", targets) == "thetaA0"
    with pytest.raises(ValueError, match="_deg"):
        strip("thetaA0_deg", targets)


def test_frozen_stage_result_rejects_a_temperature_mismatch(tmp_path):
    """`--only-*` 会把冻结 stage 与本次新采样求和，跨温度即非法（kBT 差 ~3%）。"""
    tree = ast.parse((REPO / "runabfe.py").read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_load_frozen_stage_result"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {
        "os": os, "json": json, "np": np,
        "Dict": dict, "Optional": object,
        "log": logging.getLogger("t"),
    }
    exec(compile(module, "<frozen>", "exec"), ns)
    load = ns["_load_frozen_stage_result"]

    def _write(temp_k):
        payload = {
            "stage": "vanishing", "total_delta_G": 1.0, "total_error": 0.1,
        }
        if temp_k is not None:
            payload["protocol_key"] = {"payload": {"temperature_K": temp_k}}
        p = tmp_path / f"stage2_{temp_k}.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        return str(p)

    # 同温 → 通过
    assert load(_write(300.0), "vanishing", expected_temperature_K=300.0)["stage"] == "vanishing"
    # 异温 → 拒绝
    with pytest.raises(RuntimeError, match="温度"):
        load(_write(310.0), "vanishing", expected_temperature_K=300.0)
    # 字段缺失 → 告警但不阻断（不能堵掉合法旧工件）
    assert load(_write(None), "vanishing", expected_temperature_K=300.0)["stage"] == "vanishing"
    # 不传期望温度 → 完全保持旧行为
    assert load(_write(310.0), "vanishing")["stage"] == "vanishing"


def test_traditional_total_error_no_longer_folds_in_boresch_model_error():
    """同一批工件经两条路径必须给出同一个 ±。

    `combine_binding_free_energy` 的 docstring 是唯一约定：
    "Boresch 解析释放项与 APBS 都是确定性解析量，没有独立采样方差，不并入。"
    """
    src = (REPO / "runabfe.py").read_text(encoding="utf-8")
    assert 'np.sqrt(float(cycle["total_error_kJ_mol"]) ** 2 + err_boresch**2)' not in src
    assert 'total_err_bind = float(cycle["total_error_kJ_mol"])' in src
    # 仍作为独立字段落盘，两条路径字段集对齐。
    assert src.count('"boresch_correction_error_kJ_mol"') >= 2


def test_pilot_shadow_checkpoint_interval_is_forwarded_for_both_legs():
    src = (REPO / "runabfe.py").read_text(encoding="utf-8")
    assert src.count(
        'pilot_shadow_checkpoint_interval=config.get("pilot_shadow_checkpoint_interval")'
    ) == 2


def test_apbs_note_is_restored_from_provenance():
    """值被恢复而 note 被清空 → 非零修正没有来源说明，违背 provenance 口径。"""
    src = (REPO / "runabfe.py").read_text(encoding="utf-8")
    assert 'if _flag_present("--apbs-correction-note"):' in src
    assert '"apbs_correction_note": apbs_correction_note,' in src
    assert '"apbs_correction_note": getattr(args, "apbs_correction_note", None) or "",' not in src


def test_cuda_device_spec_still_triggers_context_cleanup():
    """`CUDA:1` 以前匹配不上裸 `== "CUDA"`，显式释放整段被跳过。"""
    src = (REPO / "abfe_pipeline.py").read_text(encoding="utf-8")
    assert 'if equil_platform.upper() == "CUDA":' not in src
    assert src.count('if _split_platform_spec(equil_platform)[0].upper() == "CUDA":') == 2


def test_remd_seed_phase_is_injectable_and_defaults_unchanged():
    """phase 可注入，但**默认仍是 charging** —— 改它就是改随机流。"""
    import ibs_engine as ie
    import inspect

    sig = inspect.signature(ie.REMDManager.__init__)
    assert "seed_phase" in sig.parameters
    assert sig.parameters["seed_phase"].default is None

    src = (REPO / "ibs_engine.py").read_text(encoding="utf-8")
    assert 'self.seed_phase = str(seed_phase) if seed_phase is not None else "charging"' in src
    # 硬编码的 "charging" 必须从这三处消失。
    assert 'self._seed_for(\n                    "charging", self.seed_stage' not in src


def test_solvent_leg_receives_the_repeat_seed_contract():
    pipeline_src = (REPO / "abfe_pipeline.py").read_text(encoding="utf-8")
    core_src = (REPO / "abfe_core.py").read_text(encoding="utf-8")
    assert 'solvent_kwargs["repeat_seed"] = int(_repeat_seed)' in pipeline_src
    assert 'solvent_kwargs["leg_name"] = "solvent"' in pipeline_src
    # 构造参数必须从 pipeline_kwargs 里 pop 出来，否则会撞进 run_full_pipeline。
    assert '_repeat_seed = pipeline_kwargs.pop("repeat_seed", None)' in core_src
    assert "repeat_seed=_repeat_seed," in core_src


# ===========================================================================
# C 组：落盘 / 日志 / 可审计性
# ===========================================================================


def test_second_leg_gets_its_own_clean_log_file(tmp_path):
    """★同进程两条腿：日志必须各自分离，且不能重复。

    旧实现：logging FileHandler 每次**追加**从不摘除（第 N 条腿的行写 N 遍、
    第一条腿的文件继续收后面的行），stdout tee 又用 isinstance 短路导致第二条腿
    根本不装 tee（裸 print 全进第一条腿的文件）。
    """
    import abfe_pipeline as ap

    a = str(tmp_path / "legA.log")
    b = str(tmp_path / "legB.log")
    log = logging.getLogger("runabfe")

    saved_stdout = sys.stdout
    saved_file = ap._PROCESS_WIDE_LOG_FILE
    saved_handler = ap._PROCESS_WIDE_LOG_HANDLER
    root = logging.getLogger()
    try:
        ap._PROCESS_WIDE_LOG_FILE = None
        ap._PROCESS_WIDE_LOG_HANDLER = None

        ap._install_process_wide_log_file(a)
        print("MARK-A")
        log.warning("LOGLINE-A")

        ap._install_process_wide_log_file(b)
        print("MARK-B")
        log.warning("LOGLINE-B")
        sys.stdout.flush()

        txt_a = Path(a).read_text(encoding="utf-8")
        txt_b = Path(b).read_text(encoding="utf-8")
    finally:
        if ap._PROCESS_WIDE_LOG_HANDLER is not None:
            root.removeHandler(ap._PROCESS_WIDE_LOG_HANDLER)
            ap._PROCESS_WIDE_LOG_HANDLER.close()
        if isinstance(sys.stdout, ap._StdoutTeeToFile):
            sys.stdout.close()
        sys.stdout = saved_stdout
        ap._PROCESS_WIDE_LOG_FILE = saved_file
        ap._PROCESS_WIDE_LOG_HANDLER = saved_handler

    # 各自只有自己的行 —— 第二条腿的行不许回流到第一条腿的文件。
    assert txt_a.count("MARK-A") == 1 and "MARK-B" not in txt_a
    assert txt_a.count("LOGLINE-A") == 1 and "LOGLINE-B" not in txt_a
    # 第二条腿的文件必须真的有内容（旧实现这里几乎是空的）。
    assert txt_b.count("MARK-B") == 1 and "MARK-A" not in txt_b
    # 不许重复（旧实现 handler 累积会写 N 遍）。
    assert txt_b.count("LOGLINE-B") == 1 and "LOGLINE-A" not in txt_b


def test_walker_and_wet_seed_npz_writes_are_atomic():
    src = (REPO / "ibs_engine.py").read_text(encoding="utf-8")
    # 只剩 _atomic_save_npz 内部那一处真正的 np.savez。
    assert src.count("np.savez(handle, **arrays)") == 1
    assert "np.savez(\n        os.path.join(cache_dir" not in src
    assert "_atomic_save_npz(\n        os.path.join(cache_dir" in src
    # 读取侧要有损坏容错，别让截断 npz 卡死每次 resume。
    assert "独立端点 walker 记录不可读" in src


def test_geodesic_reports_fallback_and_dropped_edges():
    """寻径失败回退对角线以前与成功路径完全无法区分，还会被当成功结果缓存。"""
    import inspect
    import abfe_preoptimizer as pre

    sig = inspect.signature(pre.optimize_2d_geodesic_path)
    assert "diagnostics" in sig.parameters
    sig2 = inspect.signature(pre.dijkstra_monotonic_geodesic)
    assert "diagnostics" in sig2.parameters

    # 量级闸门弃边要计数：给一个处处不可通行的度量场。
    n = 4
    G = np.full((n, n, 2, 2), 1.0e9)
    diag = {}
    with pytest.raises(RuntimeError):
        pre.dijkstra_monotonic_geodesic(
            G, np.linspace(1.0, 0.0, n), np.linspace(1.0, 0.0, n), diagnostics=diag
        )
    assert diag.get("magnitude_gate_dropped_edges", 0) > 0

    # 缓存必须带上 provenance。
    pipeline_src = (REPO / "abfe_pipeline.py").read_text(encoding="utf-8")
    assert '"search_diagnostics": dict(_geodesic_diag),' in pipeline_src


def test_overlap_record_and_covariance_chain_append_atomically():
    """孤儿窗口：进了落盘统计却没进协方差链。"""
    src = (REPO / "ibs_engine.py").read_text(encoding="utf-8")
    i = src.index("_overlap_record = {")
    j = src.index("window_overlap_records.append(_overlap_record)", i)
    between = src[i:j]
    # local_results 必须先 append，overlap 记录后 append。
    assert "local_results.append({" in between


def test_split_half_marks_windows_without_evidence():
    """缺 split-half 证据的窗口以前静默 floor=0.0，看不出哪些窗口无实测。"""
    from ibs_engine import sigma_inflated_from_split_half

    full = {
        "covariance_chain_segments": [
            {"window_index": 0, "uncertainty_kJ_mol": 0.5},
            {"window_index": 1, "uncertainty_kJ_mol": 0.4},
        ]
    }
    drift = {"available": True, "per_window": [{"window_index": 0, "drift_kJ_mol": 2.0}]}
    out = sigma_inflated_from_split_half(full, drift)

    assert out["available"] is True
    rows = {r["window_index"]: r for r in out["per_window"]}
    assert rows[0]["sigma_floor_unavailable"] is False
    assert rows[1]["sigma_floor_unavailable"] is True
    assert out["windows_with_sigma_floor_unavailable"] == [1]
    assert out["n_windows_with_sigma_floor_unavailable"] == 1
    assert out["sigma_floor_coverage_complete"] is False
    # window 0 有 |漂移|=2.0 → floor=1.0 > 0.5，确实抬高。
    assert rows[0]["sigma_effective_kJ_mol"] == pytest.approx(1.0)
    # window 1 无证据 → 不抬高（数值行为不变）。
    assert rows[1]["sigma_effective_kJ_mol"] == pytest.approx(0.4)


def test_top1pct_weight_annotates_small_sample_degeneracy():
    """N<100 时该量退化成"最大单帧"，阈值 0.35 是按 N≈330-430 校准的。"""
    from ibs_engine import _ibs_reweighting_quality_diagnostics

    kt = 2.494339
    for n_frames, degenerate, n_top in ((50, True, 1), (400, False, 4)):
        u = np.zeros((2, n_frames))
        bias = np.zeros(n_frames)
        out = _ibs_reweighting_quality_diagnostics(u, bias, np.zeros(2), kt)
        assert out["top1pct_raw_weight_degenerate_max_single_frame"] is degenerate
        assert out["top1pct_raw_weight_n_top_frames"] == n_top


def test_attachment_rejects_a_sample_count_below_two():
    """`max(2, ...)` 下限会让实跑步数超过设定，且 split-half 每半只剩 1 帧。"""
    src = (REPO / "ibs_engine.py").read_text(encoding="utf-8")
    assert "n_samples = max(2, int(n_steps_per_state) // int(steps_per_sample))" not in src
    assert "_n_samples_raw < 2" in src
    assert "attachment 腿参数组合不可用" in src


def test_frozen_ligand_ligand_exception_premise_is_asserted():
    """P0-01 的前提以前零守护：frozen_ll_pairs 收集完从来没被读过。"""
    src = (REPO / "ibs_engine.py").read_text(encoding="utf-8")
    assert "def _assert_frozen_ligand_ligand_exceptions(" in src
    # 三个 decharging builder 都要调用它。
    assert src.count("_assert_frozen_ligand_ligand_exceptions(\n") == 4  # 1 def + 3 calls


def test_shadow_path_fails_closed_on_covalent_cross_group_exceptions():
    """跨组 1-4 静电在背景力与 shadow 力两边都不算 → 共价体系静默少一项。"""
    src = (REPO / "ibs_engine.py").read_text(encoding="utf-8")
    i = src.index("def _collect_shadow_cross_exclusions(")
    j = src.index("def _zero_ligand_environment_charge_in_background(", i)
    fn = src[i:j]
    assert "scaled_cross_pairs" in fn
    assert "raise RuntimeError(" in fn


def test_softcore_cv_warning_is_printed_once_not_per_state():
    src = (REPO / "ibs_engine.py").read_text(encoding="utf-8")
    i = src.index("for k, (_lc, lv) in enumerate(zip(lambdas_coul, lambdas_vdw)):")
    fn = src[i:i + 4000]
    assert "if k == 0:" in fn
    assert fn.index("if k == 0:") < fn.index("VDW custom softcore CV 不含 LJ 长程修正")


def test_overlapping_windows_docstring_matches_reality():
    from ibs_engine import generate_overlapping_windows

    assert generate_overlapping_windows(13, 6, 2) == [(0, 6), (4, 10), (7, 13)]
    src = (REPO / "ibs_engine.py").read_text(encoding="utf-8")
    assert "[(0,6), (4,10), (7,13)]" in src
    assert "[(0,6), (4,10), (8,13)]" not in src


def test_pressure_test_reads_the_constant_at_call_time():
    """模块常量被捕获为函数默认值 → 常量演进后默认值静默过期。"""
    import inspect
    import abfe_preoptimizer as pre

    sig = inspect.signature(pre.pilot_early_stop_pressure_test)
    assert sig.parameters["first_ensemble_target_intervals"].default is None


def test_preoptimizer_target_phase_is_explicit_and_defaults_to_auto():
    """auto 优先级表把 lam_coul 排在前面，vdW 阶段可能测错轴。"""
    import inspect
    import abfe_preoptimizer as pre

    sig = inspect.signature(pre.ABFEPreOptimizer.__init__)
    assert sig.parameters["target_phase"].default == "auto"


def test_error_leg_field_names_its_own_gauge():
    """字段名与内容不配对；值不能改（下游拿它配 delta_G_total 是对的），补名副其实的字段。"""
    src = (REPO / "abfe_pipeline.py").read_text(encoding="utf-8")
    assert '"error_leg_excluding_attachment_kJ_mol": float(err_leg),' in src
    assert '"error_leg_kJ_mol_pairs_with": "delta_G_total_kJ_mol",' in src
    # 历史数值语义保持不变。
    assert '"error_leg_kJ_mol": err_total,' in src

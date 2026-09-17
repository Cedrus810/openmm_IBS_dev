"""周期快照写出的**残缺** convergence.json 不得被当成「旧策略数据」。

真机 cyclod_ligand1/rep1（2026-09-15）：作业被杀在 win0 生产 200k/250k 步，
生产循环每 100 个 update 落一次 `_periodic_conv`。那段代码在
`os.path.exists(production_conv_path)` 为假时从 `{}` 起写 ⟹ 写出一份只有 8 个键、
**一个身份键都没有**的 convergence.json。重启后身份门读到
`sampling_repair_policy=None`，抛 `ExistingEnsembleRequiresRescueAudit`，
整条流水线当场死，而且该窗口**永远**进不去（那份文件不会自己消失）。

语义上这是两件事：
  · 键**存在但不同** = 盘上躺着旧变异策略采的 ensemble ⟹ fail-closed，交 rescue 审计；
  · 键**整个缺失**   = 这份记录不是一次完整的窗口收尾（周期快照 / 老格式）
                      ⟹ 就是一份无效缓存，打 WARN 重采，不该打死流水线。

⚠️ 同一判据有**两份实现**（早门 + 十门后的 raise），改一处必须改另一处。
"""
import ast
import inspect

import ibs_engine as ie
import pytest

pytestmark = pytest.mark.cpu_only


def _periodic_snapshot_like():
    """`_periodic_conv` 在文件不存在时实际写出的那 8 个键（真机盘面逐键对齐）。"""
    return {
        "stage_protocol_key": {"schema_version": 1, "sha256": "deadbeef"},
        "production_segment_protocol_version": ie.PRODUCTION_SEGMENT_PROTOCOL_VERSION,
        "production_segments": [{"start_frame": 0, "end_frame": 400}],
        "cumulative_production_steps": 200000,
        "window_data_protocol_version": ie.IBS_WINDOW_DATA_PROTOCOL_VERSION,
        "vdw_nonbonded_protocol_version": ie.VDW_NONBONDED_PROTOCOL_VERSION,
        "window_data": {"energies": {"sha256": "x", "shape": [8, 400]}},
        "loop_timing_s": {"integration_s": 128.6},
    }


_ES = {"min_steps": 0, "check_interval_steps": 0, "required_consecutive_passes": 0,
       "min_ess_ratio": 0.0, "min_absolute_ess": 0.0, "min_decorrelated_samples": 0,
       "max_delta_g_drift_kJ_mol": 0.0, "max_uncertainty_kJ_mol": 0.0}


def _gate(conv):
    return ie._resume_cached_window_gate_status(
        conv, (8, 400), [0.0] * 8, [0.9] * 8, "non_mutating_v1", 0.5, False,
        _ES, 250000, current_coion_identity=None, stage_type="vdw",
        current_sampling_score_sha256=None, current_stage_protocol_key=None)


def test_a_partial_snapshot_is_simply_an_unusable_cache():
    g = _gate(_periodic_snapshot_like())
    assert g["usable"] is False, "残缺记录当然不可复用"
    # 十门函数本身早就判对了（缺字段 ⟹ False）；坏的只是 raise 点。
    assert g["repair_policy_match"] is False


def test_missing_policy_key_does_not_raise_rescue_audit():
    """两个 raise 点都必须先判「键存在」才 fail-closed。"""
    src = inspect.getsource(ie.IBSWindowManagerDualLambda.run_all_windows)
    tree = ast.parse(src.lstrip())
    raises = [n for n in ast.walk(tree)
              if isinstance(n, ast.Raise)
              and "ExistingEnsembleRequiresRescueAudit" in ast.unparse(n)
              and "sampling_repair_policy" in ast.unparse(n)]
    assert len(raises) == 2, f"raise 点数量变了（{len(raises)}），两份实现必须同时守"
    # 早门
    assert "cached_policy_early is not None" in src
    # 十门之后那处
    assert 'and cached_conv.get("sampling_repair_policy") is not None' in src


def test_a_real_policy_mismatch_still_fails_closed():
    """**不许放宽**：键存在但不同，仍然是旧策略数据，仍然 fail-closed。"""
    conv = dict(_periodic_snapshot_like(), sampling_repair_policy="legacy_mutating")
    assert _gate(conv)["repair_policy_match"] is False
    src = inspect.getsource(ie.IBSWindowManagerDualLambda.run_all_windows)
    # 判据是 `is not None`，不是被删掉或改成恒真
    assert "cached_policy_early != repair_policy" in src


def test_the_periodic_snapshot_still_lacks_the_identity_block():
    """**已知缺口的钉子**：根因在写侧，读侧只是不再因此致命。

    `_periodic_conv` 在文件不存在时从 `{}` 起写 ⟹ 写出的记录没有任何身份键
    ⟹ 它永远不可能被当成可复用缓存，被杀时那一段生产帧仍然会丢。
    读侧已修（不再打死流水线），但进度**没有**被保住。

    这条断言故意是"缺口还在"的形状：**补上身份块之后它会红**，
    那正是提醒——届时请把这个文件一起更新，并把 `usable=True` 的新路径钉住。
    """
    src = inspect.getsource(ie.IBSWindowManagerDualLambda.run_all_windows)
    block = src.split("_periodic_conv = {}")[1].split("_atomic_write_json(")[0]
    missing = [k for k in ("sampling_repair_policy", "lambdas_vdw",
                           "wca_accounting_version", "ibs_bias_protocol_version")
               if f'"{k}"' not in block]
    assert missing == ["sampling_repair_policy", "lambdas_vdw",
                       "wca_accounting_version", "ibs_bias_protocol_version"], (
        f"周期快照的身份块变了（现在缺 {missing}）—— 缺口被补上或改动了，"
        "请更新本测试并为新行为补钉子")

"""P2-14 回归：`final_results.json` 不得谎报 LJ 长程尾项已应用。

旧行为：`ABFEPipeline.compute_final_results` 里
`lj_long_range_dispersion_correction.applied` 是**裸字面量 `True`**，整个 dict
literal 没有任何分支，后面那段 ~30 行 note 还用散文再断言一遍"修正已生效并已折进
total_delta_G_complex_kJ_mol"。

而 `ibs_engine.build_ibs_dual_system` 对 DEXP 明确不附加：
`ibs_wrapper.lj_tail_lrc_coeff_kj_mol = None`，三个消费者
（`IBSSampler._lj_tail_correction_kj_mol`、fixed-H overlap 探针、production）
都 short-circuit 到零。所以 DEXP 运行产出的 JSON 在机器可读字段和人读说明上
同时是错的。

修法刻意**不是**在报告侧另写一个 `potential_type != "dexp"`：那样两处判据会分叉。
生产者和报告者共用同一个谓词 `ibs_lj_tail_lrc_is_applicable`，DEXP 的解析尾项
公式一旦被验证/替换，只需要改那一处，行为和报告会一起动。
"""

import ast
from pathlib import Path

import pytest

from ibs_engine import (
    ibs_lj_tail_lrc_inapplicable_reason,
    ibs_lj_tail_lrc_is_applicable,
)

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "potential_type, expected",
    [
        ("softcore", True),
        ("ace", True),
        ("beutler", True),
        (None, True),
        ("dexp", False),
        ("DEXP", False),      # 大小写不敏感
        ("  dexp  ", False),  # 前后空白
    ],
)
def test_applicability_predicate(potential_type, expected):
    assert ibs_lj_tail_lrc_is_applicable(potential_type) is expected


def test_reason_is_empty_exactly_when_applicable():
    assert ibs_lj_tail_lrc_inapplicable_reason("softcore") == ""
    reason = ibs_lj_tail_lrc_inapplicable_reason("dexp")
    assert reason and "dexp" in reason


def test_producer_uses_the_shared_predicate_not_an_inline_comparison():
    """生产者侧必须调谓词。写回 `potential_type == "dexp"` 会让两处再次分叉。"""
    source = (REPO_ROOT / "ibs_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="ibs_engine.py")

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_ibs_dual_system":
            target = node
            break
    assert target is not None, "找不到 build_ibs_dual_system"

    called = {
        n.func.id
        for n in ast.walk(target)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "ibs_lj_tail_lrc_is_applicable" in called, (
        "build_ibs_dual_system 必须用共享谓词决定要不要附加 LRC"
    )

    # 同一函数体里不得再出现裸的 potential_type == "dexp" 比较来控制 LRC。
    for node in ast.walk(target):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name):
            if node.left.id != "potential_type":
                continue
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and comparator.value == "dexp":
                    pytest.fail(
                        "build_ibs_dual_system 里又出现了内联的 "
                        'potential_type == "dexp" 比较；LRC 的适用性判据只能有一处'
                    )


def test_reporter_does_not_hardcode_applied_true():
    """报告侧不得再出现 `"applied": True` 这种裸字面量。"""
    source = (REPO_ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="abfe_pipeline.py")

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and key.value == "applied"):
                continue
            if isinstance(value, ast.Constant) and value.value is True:
                offenders.append(getattr(value, "lineno", -1))
    assert not offenders, (
        f"abfe_pipeline.py 第 {offenders} 行又把 \"applied\" 写成了字面量 True；"
        "它必须来自生产者（traditional_lj_lrc metadata 或 "
        "ibs_lj_tail_lrc_is_applicable）"
    )


def _lrc_block(tmp_path, monkeypatch, potential_type, sampling_results, scheme):
    """跑一次真正的 compute_final_results，取出 LRC 段。

    空 System 足以记录约束身份，不需要为报告测试构建 Context。
    """
    from openmm import unit

    import abfe_pipeline as ap

    pipeline = ap.ABFEPipeline.__new__(ap.ABFEPipeline)
    pipeline._last_run_config = {"potential_type": potential_type}
    pipeline.output_dir = str(tmp_path)
    pipeline.system = ap.openmm.System()
    pipeline.topology = None
    pipeline.positions = None
    pipeline.ligand_indices = []
    pipeline.temperature = 300.0 * unit.kelvin
    pipeline._log = lambda *a, **k: None
    monkeypatch.setattr(ap, "_collect_pipeline_provenance", lambda **kw: {})

    final = pipeline.compute_final_results(
        sampling_results=sampling_results,
        correction_results={"delta_g_rest": 0.0, "error": 0.0},
        decoupling_scheme=scheme,
    )
    return final["lj_long_range_dispersion_correction"]


def test_dexp_run_reports_not_applied(tmp_path, monkeypatch):
    """端到端：potential_type='dexp' → applied 必须 False 且 note 打头就说明未应用。"""
    block = _lrc_block(
        tmp_path, monkeypatch, "dexp",
        {"total_delta_G": 1.0, "total_error": 0.1}, "dual_lambda",
    )
    assert block["applied"] is False
    assert block["applicable"] is False
    assert block["potential_type"] == "dexp"
    assert block["not_applied_reason"]
    assert block["status"].startswith("not_applied")
    assert block["note"].startswith("NOT APPLIED"), (
        "人读说明也不能继续断言修正已生效——机器字段和散文必须一起改口"
    )


def test_softcore_run_still_reports_applied(tmp_path, monkeypatch):
    block = _lrc_block(
        tmp_path, monkeypatch, "softcore",
        {"total_delta_G": 1.0, "total_error": 0.1}, "dual_lambda",
    )
    assert block["applied"] is True
    assert block["not_applied_reason"] is None
    assert not block["note"].startswith("NOT APPLIED")


def test_traditional_path_truth_beats_the_predicate(tmp_path, monkeypatch):
    """传统 REMD 路径有自己的 metadata；它才是那条路径的真相，优先级更高。

    这里刻意让两者冲突：potential_type='softcore' 时谓词会说 True，但生产者
    （TraditionalMBARAnalyzer 的 lj_lrc_metadata）说没算成。报告必须听生产者的。
    """
    block = _lrc_block(
        tmp_path, monkeypatch, "softcore",
        {
            "total_delta_G": 1.0,
            "total_error": 0.1,
            "diagnostics": {
                "traditional_lj_lrc": {"applicable": False, "applied": False},
            },
        },
        "single_lambda",
    )
    assert block["applied"] is False
    assert block["truth_source"] == "traditional_mbar_analyzer_lj_lrc_metadata"

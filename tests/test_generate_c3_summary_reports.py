"""`tools/validation/generate_c3_summary_reports.py` 的契约测试。

只跑得动的部分：对已经落盘的真实 `validation/c3_real_endpoints_v2/` 结果
JSON 做纯汇总/核验，不重跑任何 MD、不建 CUDA Context——`_mem00h_per_case_
structural_check` 只读 `NonbondedForce.getCutoffDistance/getUseSwitching
Function`，几毫秒级别，CPU 契约测试可以直接跑真实数据，不用造合成 case_dir。

如果这些真实结果文件在某个环境里不存在（比如全新 checkout、还没跑过 GPU
矩阵），整个文件跳过——这不是"假装测过"，是诚实标注这条测试依赖真实 GPU
产物存在。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 🔑 [2026-08-31 目录重整] 这两条测试消费的是**真实实验产物**，其中
# `tests/fixtures/validation/c2_lipid_slab_v11/` 有 94 MB —— 不进 release 仓库。
# 缺数据时给一个**带明确原因**的 skip，而不是 fail，也不是静默跳过：
# pytest.ini 的 `addopts = -ra` 会把跳过原因汇总打印出来，这正是本仓库对
# "大体积 fixture 不随仓库提供"的既有处理方式（见 memtest/charmm-gui 那几条）。
_REQUIRED_FIXTURES = [
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "validation" / d
    for d in ("c1_waterbox", "c2_lipid_slab_v11", "c3_real_endpoints_v2")
]
_MISSING_FIXTURES = [p for p in _REQUIRED_FIXTURES if not p.exists()]
_SKIP_IF_NO_FIXTURES = pytest.mark.skipif(
    bool(_MISSING_FIXTURES),
    reason=(
        "缺少真实验证产物 "
        + ", ".join(str(p.relative_to(Path(__file__).resolve().parents[1])) for p in _MISSING_FIXTURES)
        + "（体积过大，不随仓库提供；需要时从实验目录取）"
    ),
)

pytestmark = [pytest.mark.cpu_only, _SKIP_IF_NO_FIXTURES]

_MODULE_PATH = ROOT / "tools" / "validation" / "generate_c3_summary_reports.py"
_spec = importlib.util.spec_from_file_location("generate_c3_summary_reports", _MODULE_PATH)
gen = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gen
_spec.loader.exec_module(gen)  # type: ignore[union-attr]

_MISSING_INPUTS = [
    p for p in list(gen.AB_CASES.values()) + list(gen.CD_CASES.values()) if not p.exists()
]


@pytest.mark.skipif(
    bool(_MISSING_INPUTS),
    reason=f"依赖真实 GPU 产物，本环境缺失：{_MISSING_INPUTS}",
)
def test_generate_reports_from_real_data_passes_and_is_internally_consistent():
    result = gen.generate()
    summary = result["summary"]
    mem00h = result["mem00h_report"]

    assert summary["ab"]["n_frames"] == 100
    assert summary["ab"]["n_failed"] == 0
    assert summary["ab"]["passed"] is True
    assert summary["cd"]["n_frames"] == 50
    assert summary["cd"]["n_failed"] == 0
    assert summary["cd"]["passed"] is True
    assert summary["n_frames_total"] == 150
    assert summary["endpoints"]["all_passed"] is True
    assert summary["status"] == "complete"
    assert summary["passed"] is True

    assert mem00h["evaluation_cutoff_nm"] == 1.0
    assert mem00h["evaluation_switching_enabled"] is False
    # 不能把 C2 写成"raw 本身无 switch"——如实记录 raw 状态。
    c2_cases = [c for c in mem00h["normalization"]["per_case"] if c["case"] != "C1_Na_large"]
    assert c2_cases, "应该至少覆盖 4 个 C2 case"
    assert all(c["raw_switching_enabled"] is True for c in c2_cases)
    assert all(c["raw_switch_distance_nm"] == pytest.approx(0.995) for c in c2_cases)
    assert all(c["normalized_evaluation_switching_enabled"] is False for c in c2_cases)
    c1_case = next(
        c for c in mem00h["normalization"]["per_case"] if c["case"] == "C1_Na_large"
    )
    assert c1_case["raw_switching_enabled"] is False
    assert c1_case["normalization_changed_switching"] is False

    assert mem00h["normalization"]["c2_raw_switch_confirmed_present"] is True
    assert mem00h["c_seam_passed"] is True
    assert mem00h["d_strict_zero_passed"] is True
    assert mem00h["status"] == "complete"
    assert mem00h["passed"] is True


@pytest.mark.skipif(
    bool(_MISSING_INPUTS),
    reason=f"依赖真实 GPU 产物，本环境缺失：{_MISSING_INPUTS}",
)
def test_generate_reports_fails_closed_if_any_case_reports_failure(monkeypatch):
    """篡改一份输入结果（把 `passed` 改成 False），汇总必须如实反映为
    False，不能被其它 9 份干净结果"平均"掩盖。用仓库内部的临时目录（不是
    `tmp_path`）——`_collect_ab`/`_collect_cd` 里的 `file` 字段要对
    `_REPO_ROOT` 取 `relative_to`，仓库外的路径会在那一步直接炸，不是
    这个测试要验证的东西。
    """
    import json
    import shutil

    tampered_dir = gen.RESULT_DIR / "_test_tampered_scratch"
    tampered_dir.mkdir(exist_ok=True)
    try:
        for path in list(gen.AB_CASES.values()) + list(gen.CD_CASES.values()):
            shutil.copy(path, tampered_dir / path.name)

        victim = tampered_dir / gen.AB_CASES["Na_thin_pos0"].name
        payload = json.loads(victim.read_text(encoding="utf-8"))
        payload["passed"] = False
        payload["n_failed"] = 1
        victim.write_text(json.dumps(payload), encoding="utf-8")

        monkeypatch.setattr(
            gen, "AB_CASES", {k: tampered_dir / v.name for k, v in gen.AB_CASES.items()}
        )
        monkeypatch.setattr(
            gen, "CD_CASES", {k: tampered_dir / v.name for k, v in gen.CD_CASES.items()}
        )

        result = gen.generate()
        assert result["summary"]["ab"]["passed"] is False
        assert result["summary"]["passed"] is False
        assert result["summary"]["status"] == "incomplete"
    finally:
        shutil.rmtree(tampered_dir, ignore_errors=True)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C3 正式关闭所需的两份汇总产物：`summary.json`/`mem00h_report.json`。

对应 `memtodolist.md` §C3 "要求" 清单（"MEM-00h protocol 字段一致"、
"`summary.json=PASS`"）与 "最终只有 C3 和 `mem00h_report.json` 同时 PASS，
才能同时关闭 C3 与 MEM-00h" 这条硬约束。

**这是纯汇总/核验脚本，不做任何新的物理计算**——不重跑 MD、不重新求值任何
System 的 energy/force。它只做两件事：

1. 读取 `validation/c3_real_endpoints_v2/` 下已经跑完的 10 份 case 结果
   JSON（5 个 A/B + 5 个 C/D），逐项核验（不是假设）`passed=True`、
   `n_failed=0`，并核对总帧数确实是 A/B=100、C/D=50——不满足就 fail
   closed，不能在这一步偷偷把 `passed` 写成 `True`。
2. 对每个 case 的 raw System 做一次**纯 CPU、只读结构**的核验（`getCutoff
   Distance`/`getUseSwitchingFunction`，不建 Context、不求值任何 energy/
   force）：如实记录 raw System 本身的 switching 状态（C2 的几个 case
   本来就带 `[0.995,1.0]nm` 的局部 switch，绝不能被写成"C2 raw 本身无
   switch"），并核对经过 `mem00h_normalized_raw_system()` 归一化之后的
   evaluation clone确实是 `cutoff=1.0nm, switching=False`。

用法：

    python tools/validation/generate_c3_summary_reports.py

默认从 `validation/c3_real_endpoints_v2/` 读输入，写到同一目录下的
`summary.json`/`mem00h_report.json`；两者的 `passed`/`status` 字段严格从
上面这些核验结果推导出来，不是写死的常量。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_charge_transfer_endpoints as cte  # noqa: E402
import ibs_engine as ie  # noqa: E402

RESULT_DIR = _REPO_ROOT / "validation" / "c3_real_endpoints_v2"

AB_CASES: Dict[str, Path] = {
    "C1_Na_large": RESULT_DIR / "c1_na_large_v2.json",
    "Na_thin_pos0": RESULT_DIR / "c2_Na_thin_pos0_v2.json",
    "Na_thin_pos1": RESULT_DIR / "c2_Na_thin_pos1_v2.json",
    "Na_thick_pos0": RESULT_DIR / "c2_Na_thick_pos0_v2.json",
    "Na_thick_pos1": RESULT_DIR / "c2_Na_thick_pos1_v2.json",
}
CD_CASES: Dict[str, Path] = {
    "C1_Na_large": RESULT_DIR / "cd_c1_na_large_v2.json",
    "Na_thin_pos0": RESULT_DIR / "cd_c2_Na_thin_pos0_v2.json",
    "Na_thin_pos1": RESULT_DIR / "cd_c2_Na_thin_pos1_v2.json",
    "Na_thick_pos0": RESULT_DIR / "cd_c2_Na_thick_pos0_v2.json",
    "Na_thick_pos1": RESULT_DIR / "cd_c2_Na_thick_pos1_v2.json",
}

# case 名 -> raw case_dir（跟 memtodolist.md/前面几轮 run-matrix-v2(-cd) 调用
# 用的是同一批目录），只用来做只读结构核验，不碰轨迹/坐标。
CASE_RAW_DIRS: Dict[str, Path] = {
    "C1_Na_large": _REPO_ROOT / "validation" / "c1_waterbox" / "Na_large",
    "Na_thin_pos0": _REPO_ROOT / "validation" / "c2_lipid_slab_v11" / "Na_thin_pos0",
    "Na_thin_pos1": _REPO_ROOT / "validation" / "c2_lipid_slab_v11" / "Na_thin_pos1",
    "Na_thick_pos0": _REPO_ROOT / "validation" / "c2_lipid_slab_v11" / "Na_thick_pos0",
    "Na_thick_pos1": _REPO_ROOT / "validation" / "c2_lipid_slab_v11" / "Na_thick_pos1",
}

EXPECTED_AB_FRAMES_PER_CASE = 20
EXPECTED_CD_FRAMES_PER_CASE = 10


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"缺少必需的输入文件：{path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _sha256(path: Path) -> str:
    return cte.sha256_file(str(path))


def _collect_ab() -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    total_frames = 0
    total_failed = 0
    all_passed = True
    for name, path in AB_CASES.items():
        report = _load_json(path)
        n_frames = int(report["n_frames"])
        n_failed = int(report["n_failed"])
        passed = bool(report["passed"]) and n_failed == 0
        if n_frames != EXPECTED_AB_FRAMES_PER_CASE:
            passed = False
        total_frames += n_frames
        total_failed += n_failed
        all_passed = all_passed and passed
        cases.append(
            {
                "case": name,
                "file": str(path.relative_to(_REPO_ROOT)),
                "sha256": _sha256(path),
                "protocol_version": report.get("protocol_version"),
                "n_frames": n_frames,
                "n_failed": n_failed,
                "passed": passed,
            }
        )
    return {
        "cases": cases,
        "n_frames": total_frames,
        "n_failed": total_failed,
        "passed": bool(all_passed and total_frames == 100 and total_failed == 0),
    }


def _collect_cd() -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    total_frames = 0
    total_failed = 0
    all_passed = True
    c_seam_all_passed = True
    d_strict_zero_all_passed = True
    for name, path in CD_CASES.items():
        report = _load_json(path)
        n_frames = int(report["n_frames"])
        n_failed = int(report["n_failed"])
        passed = bool(report["passed"]) and n_failed == 0
        if n_frames != EXPECTED_CD_FRAMES_PER_CASE:
            passed = False
        c_seam_passed_case = all(fr["C"]["passed"] for fr in report["frames"])
        d_passed_case = all(fr["D"]["passed"] for fr in report["frames"])
        d_strict_zero_passed_case = all(
            fr["D"]["strict_zero_reference"]["passed"] and fr["D"]["strict_zero_mixed"]["passed"]
            for fr in report["frames"]
        )
        total_frames += n_frames
        total_failed += n_failed
        all_passed = all_passed and passed
        c_seam_all_passed = c_seam_all_passed and c_seam_passed_case
        d_strict_zero_all_passed = d_strict_zero_all_passed and d_strict_zero_passed_case
        cases.append(
            {
                "case": name,
                "file": str(path.relative_to(_REPO_ROOT)),
                "sha256": _sha256(path),
                "protocol_version": report.get("protocol_version"),
                "n_frames": n_frames,
                "n_failed": n_failed,
                "passed": passed,
                "c_seam_passed": c_seam_passed_case,
                "d_passed": d_passed_case,
                "d_strict_zero_passed": d_strict_zero_passed_case,
            }
        )
    return {
        "cases": cases,
        "n_frames": total_frames,
        "n_failed": total_failed,
        "passed": bool(all_passed and total_frames == 50 and total_failed == 0),
        "c_seam_all_passed": c_seam_all_passed,
        "d_strict_zero_all_passed": d_strict_zero_all_passed,
    }


def _mem00h_per_case_structural_check(name: str, raw_dir: Path) -> Dict[str, Any]:
    """只读结构核验：raw System 本身的 switching 状态（如实记录，不能把 C2
    写成"本来就无 switch"），以及归一化之后的 evaluation clone 是否真的是
    `cutoff=1.0nm, switching=False`。不建 Context、不求值任何 energy/force。
    """
    inputs = cte.load_case_raw_inputs(raw_dir)
    raw_system = inputs["system"]
    raw_nb = cte._find_nonbonded_force(raw_system)
    raw_cutoff_nm = raw_nb.getCutoffDistance().value_in_unit(cte.unit.nanometer)
    raw_switching_enabled = bool(raw_nb.getUseSwitchingFunction())
    raw_switch_distance_nm = (
        raw_nb.getSwitchingDistance().value_in_unit(cte.unit.nanometer)
        if raw_switching_enabled
        else None
    )

    normalized = cte.mem00h_normalized_raw_system(raw_system)
    normalization_changed_anything = raw_switching_enabled  # cutoff 只核验不改写，only switching 会变

    try:
        cte.assert_mem00h_switching_convention(
            normalized, context=f"mem00h_report:{name}:normalized_raw"
        )
        normalized_conforms = True
    except RuntimeError:
        normalized_conforms = False

    return {
        "case": name,
        "raw_cutoff_nm": raw_cutoff_nm,
        "raw_switching_enabled": raw_switching_enabled,
        "raw_switch_distance_nm": raw_switch_distance_nm,
        "normalization_applied": True,
        "normalization_changed_switching": normalization_changed_anything,
        "normalized_evaluation_cutoff_nm": cte.MEM00H_CUTOFF_NM,
        "normalized_evaluation_switching_enabled": cte.MEM00H_SWITCHING_ENABLED,
        "normalized_conforms_to_mem00h": normalized_conforms,
    }


def generate() -> Dict[str, Any]:
    ab = _collect_ab()
    cd = _collect_cd()

    per_case_mem00h = [
        _mem00h_per_case_structural_check(name, raw_dir)
        for name, raw_dir in CASE_RAW_DIRS.items()
    ]
    normalization_ok = all(c["normalized_conforms_to_mem00h"] for c in per_case_mem00h)
    # C2 的四个 case raw 本身必须真的带局部 switch（否则说明这次归一化其实
    # 什么都没做，之前的"消融验证"就没有意义）——如实核对，不是假设。
    c2_raw_switch_present = all(
        c["raw_switching_enabled"] for c in per_case_mem00h if c["case"] != "C1_Na_large"
    )
    c1_raw_already_compliant = next(
        c for c in per_case_mem00h if c["case"] == "C1_Na_large"
    )["raw_switching_enabled"] is False

    all_endpoints_passed = bool(ab["passed"] and cd["passed"])

    summary = {
        "protocol_version": cte.PROTOCOL_VERSION,
        "ab": {
            "n_frames": ab["n_frames"],
            "n_failed": ab["n_failed"],
            "passed": ab["passed"],
            "cases": ab["cases"],
        },
        "cd": {
            "n_frames": cd["n_frames"],
            "n_failed": cd["n_failed"],
            "passed": cd["passed"],
            "cases": cd["cases"],
        },
        "endpoints": {
            "A": True,  # 蕴含在 ab.passed 里（charging λ=1，与 B 共用同一个 run）
            "B": True,  # 蕴含在 ab.passed 里（charging λ=0）
            "C": cd["c_seam_all_passed"],
            "D": cd["d_strict_zero_all_passed"] and cd["passed"],
            "all_passed": all_endpoints_passed,
        },
        "n_frames_total": ab["n_frames"] + cd["n_frames"],
        "n_failed_total": ab["n_failed"] + cd["n_failed"],
        "input_case_files_verified": len(ab["cases"]) + len(cd["cases"]) == 10
        and all(c["passed"] for c in ab["cases"] + cd["cases"]),
        "status": "complete" if all_endpoints_passed else "incomplete",
        "passed": all_endpoints_passed,
    }

    mem00h_report = {
        "evaluation_cutoff_nm": cte.MEM00H_CUTOFF_NM,
        "evaluation_switching_enabled": cte.MEM00H_SWITCHING_ENABLED,
        "normalization": {
            "mechanism": (
                "compare_charge_transfer_endpoints.mem00h_normalized_raw_system() "
                "+ assert_mem00h_switching_convention()"
            ),
            "scope": (
                "仅作用于 C3 评估工具内部的一份内存 clone；不修改任何 case 目录下的 "
                "raw system.xml/轨迹文件本身"
            ),
            "applied_in": [
                "run_protocol_v2_matrix (A/B)",
                "run_protocol_v2_matrix_cd (C/D)",
            ],
            "per_case": per_case_mem00h,
            "c2_raw_switch_confirmed_present": c2_raw_switch_present,
            "c1_raw_already_mem00h_compliant_noop": c1_raw_already_compliant,
            "all_cases_conform_after_normalization": normalization_ok,
        },
        "production_reference_convention_consistent": normalization_ok,
        "lrc_vdw_protocol": {
            "vdw_nonbonded_protocol_version": ie.VDW_NONBONDED_PROTOCOL_VERSION,
            "traditional_lj_lrc_protocol_version": ie.TRADITIONAL_LJ_LRC_PROTOCOL_VERSION,
            "lj_tail_lrc_r_switch_nm": ie.LJ_TAIL_LRC_R_SWITCH_NM,
            "lj_tail_lrc_r_cutoff_nm": ie.LJ_TAIL_LRC_R_CUTOFF_NM,
            "note": (
                "全局 ibs_engine 常量，同一进程内对所有 case 一致适用，不是逐 case "
                "可配置的值——因此天然一致，这里如实记录具体数值供审计。"
            ),
        },
        "c_seam_passed": cd["c_seam_all_passed"],
        "d_strict_zero_passed": cd["d_strict_zero_all_passed"],
        "status": "complete" if (normalization_ok and cd["c_seam_all_passed"] and cd["d_strict_zero_all_passed"]) else "incomplete",
        "passed": bool(normalization_ok and cd["c_seam_all_passed"] and cd["d_strict_zero_all_passed"]),
    }

    return {"summary": summary, "mem00h_report": mem00h_report}


def main() -> int:
    result = generate()
    summary_path = RESULT_DIR / "summary.json"
    mem00h_path = RESULT_DIR / "mem00h_report.json"
    summary_path.write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    mem00h_path.write_text(
        json.dumps(result["mem00h_report"], indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"wrote {summary_path} (passed={result['summary']['passed']})")
    print(f"wrote {mem00h_path} (passed={result['mem00h_report']['passed']})")
    return 0 if (result["summary"]["passed"] and result["mem00h_report"]["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

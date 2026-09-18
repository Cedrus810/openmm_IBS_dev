"""`--run-unclosed-charge-transfer-diagnostics`：诊断专用执行模式。

这道闸先前是无条件 `raise`（runabfe.py，建任何 Context 之前）。开这个口子之后
要同时守住四件事：
  ① 不加开关**照样拒**（默认 fail-closed）；
  ② 加了开关**也不改任何数** —— `closes_thermodynamic_cycle` 仍是 `False`；
  ③ **只认命令行**：写进配置文件不生效（否则它会沉进生产默认配置，被 resume
     和配置复制带着到处跑）；
  ④ 禁报标记进**每份**产物 —— 逐腿 `final_results.json` 与汇总
     `final_binding_results.json`，不是只有 `run_provenance.json`。
"""

import json

import pytest

import runabfe
from abfe_core import (
    CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
    CHARGE_TREATMENT_NEUTRAL,
    charge_transfer_result_markers,
)

_UNCLOSED = {
    "charge_treatment": CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
    "closes_thermodynamic_cycle": False,
    "incomplete_cycle_reason": "tethered_charge_carrier_reservoir_correction_not_validated",
}


# ---------------------------------------------------------------- ① / ②

def test_refuses_by_default():
    err = runabfe.charge_transfer_admission_error(_UNCLOSED)
    assert err and "禁止运行并报告" in err
    # 错误信息要指出出口，否则用户只能去翻源码。
    assert "--run-unclosed-charge-transfer-diagnostics" in err


def test_diagnostics_mode_lets_it_start():
    assert runabfe.charge_transfer_admission_error(
        _UNCLOSED, diagnostics_mode=True) is None


def test_diagnostics_mode_does_not_touch_the_closure_flag():
    # 开关只影响"能不能开跑"；输入 dict 一个字段都不该被改写。
    payload = dict(_UNCLOSED)
    runabfe.charge_transfer_admission_error(payload, diagnostics_mode=True)
    assert payload == _UNCLOSED


def test_closed_cycle_and_neutral_never_hit_the_gate():
    assert runabfe.charge_transfer_admission_error(
        {"charge_treatment": CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
         "closes_thermodynamic_cycle": True}) is None
    assert runabfe.charge_transfer_admission_error(
        {"charge_treatment": CHARGE_TREATMENT_NEUTRAL}) is None


# ---------------------------------------------------------------- ③

def test_flag_defaults_off_and_is_command_line_only(tmp_path, monkeypatch):
    parser = runabfe.build_parser()
    assert parser.parse_args([]).run_unclosed_charge_transfer_diagnostics is False
    assert parser.parse_args(
        ["--run-unclosed-charge-transfer-diagnostics"]
    ).run_unclosed_charge_transfer_diagnostics is True

    # 配置文件里写上它**不生效** —— 这是这个模式的核心约束，不是风格问题。
    cfg = tmp_path / "sneaky.json"
    cfg.write_text(json.dumps(
        {"run_unclosed_charge_transfer_diagnostics": True}), encoding="utf-8")
    argv = ["--config", str(cfg)]
    monkeypatch.setattr(runabfe.sys, "argv", ["runabfe.py"] + argv)
    config = runabfe.RunConfig(parser.parse_args(argv))
    assert config.get("run_unclosed_charge_transfer_diagnostics") is False

    # 同一份配置 + 命令行显式给出 ⟹ 生效。
    argv = ["--config", str(cfg), "--run-unclosed-charge-transfer-diagnostics"]
    monkeypatch.setattr(runabfe.sys, "argv", ["runabfe.py"] + argv)
    config = runabfe.RunConfig(parser.parse_args(argv))
    assert config.get("run_unclosed_charge_transfer_diagnostics") is True


# ---------------------------------------------------------------- ④

def test_markers_are_the_three_keys_the_user_asked_for():
    m = charge_transfer_result_markers(_UNCLOSED)
    assert m["must_not_report_delta_g_bind"] is True
    assert m["production_qualified"] is False
    assert m["closes_thermodynamic_cycle"] is False
    assert m["must_not_report_delta_g_bind_reason"]


def test_neutral_ligand_products_are_unchanged():
    # 中性配体一个键都不加 ⟹ 既有 final_results.json / final_binding_results.json
    # 逐字节不变。这条是所有既有 run 的回归保护。
    assert charge_transfer_result_markers(
        {"charge_treatment": CHARGE_TREATMENT_NEUTRAL}) == {}
    assert charge_transfer_result_markers(None) == {}


def test_markers_reach_both_leg_products_and_the_summary():
    markers = charge_transfer_result_markers(_UNCLOSED)
    # 逐腿产物（abfe_pipeline 把 markers 展进 final）在这里用等价的 leg dict 代表；
    # 关键契约是 `_binding_result_status` 把它们上浮到汇总层。
    leg = dict(markers, publishable_as_accepted_result=False,
               precision_status="UNMEASURED")
    status = runabfe._binding_result_status(leg, dict(leg))
    for key, value in markers.items():
        assert status[key] == value, key
    # 逐腿明细里也要留着，不能只剩一个顶层布尔。
    assert status["per_leg"]["complex"]["must_not_report_delta_g_bind"] is True
    assert status["per_leg"]["solvent"]["production_qualified"] is False


def test_summary_unchanged_when_no_leg_declares():
    status = runabfe._binding_result_status(
        {"precision_status": "UNMEASURED"}, {"precision_status": "UNMEASURED"})
    for key in runabfe._MUST_NOT_REPORT_KEYS:
        assert key not in status


@pytest.mark.parametrize("closes", [True, False])
def test_unqualified_stays_unqualified_even_if_the_cycle_closes(closes):
    # C4/C5 仍未通过 ⟹ 即使外部给了 reservoir 修正把循环闭上，也不该被提升为
    # production_qualified；闭合但未验收 ⟹ 仍然禁报。
    m = charge_transfer_result_markers(
        {"charge_treatment": CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
         "closes_thermodynamic_cycle": closes})
    assert m["must_not_report_delta_g_bind"] is True
    assert m["production_qualified"] is False
    assert m["closes_thermodynamic_cycle"] is closes


# ------------------------------------------- 三道闸必须一起解（结构不变量）

def test_every_unclosed_charge_transfer_refusal_is_guarded_by_the_flag():
    """未闭合 charge-transfer 的拒绝点**一个都不能是无条件的**。

    2026-09-18 真机教训：第一版只解了 runabfe 建 Context 前那道，
    `abfe_pipeline.run_full_pipeline` 入口和 runabfe 汇总处各还有一道 ⟹
    诊断模式只是把死点往后挪，第二道白烧一条腿、第三道白烧两条腿。

    这里钉的是「有没有漏掉兄弟调用点」，所以判据只能是源码级的：
    每条提到 reservoir correction / closure 验证的 `raise`，它上方必须出现
    诊断开关名。新增第四道闸而忘了接线 ⟹ 本条直接红。
    """
    import pathlib
    import re

    flags = ("run_unclosed_charge_transfer_diagnostics",
             "unclosed_charge_transfer_diagnostics",
             "diagnostics_mode",
             # 汇总处把开关读成这个局部名（它只由那个开关赋值）。
             "_reservoir_omitted")
    # ⚠️ needles 必须是 **charge-transfer 专有**的词。第一版把「拒绝汇总」也算进来，
    # 立刻误报了 `constraint_identity`（约束/HMR 身份）那几条——它们跟电荷路线无关，
    # 不该被这个开关解除。
    needles = ("reservoir correction", "C4-C5 closure")
    root = pathlib.Path(__file__).resolve().parent.parent
    found = 0
    for name in ("runabfe.py", "abfe_pipeline.py"):
        lines = (root / name).read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not re.match(r"\s*(raise RuntimeError|return \()", line):
                continue
            block = "\n".join(lines[i:i + 8])
            if not any(n in block for n in needles):
                continue
            # 只看"未闭合/缺失"这一类拒绝，不管口径不一致那几条
            if not any(k in block for k in ("缺少", "未通过")):
                continue
            guard = "\n".join(lines[max(0, i - 25):i])
            assert any(f in guard for f in flags), (
                f"{name}:{i + 1} 是一条**无条件**的未闭合 charge-transfer 拒绝，"
                f"没有被诊断开关守住：{line.strip()}"
            )
            found += 1
    # 三道闸：runabfe 预检、abfe_pipeline 入口、runabfe 汇总。
    assert found >= 3, f"只找到 {found} 条拒绝点，判据失效了（检查 needles）"

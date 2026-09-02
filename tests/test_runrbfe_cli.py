"""`runrbfe.py` 的 R0 CLI（RBFE 计划 §8 的 R0 交付物「CLI validate」）。

R0 验收：错误输入被拒绝；合成两腿数据的符号／单位／误差传播正确；**不启动 GPU**。
本文件全部走内存与 tmp_path，不建任何 System。
"""

from __future__ import annotations

import copy
import json

import pytest

import rbfe_core as rc
import runrbfe


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


@pytest.fixture
def template():
    return copy.deepcopy(runrbfe.TEMPLATE)


# ---------------------------------------------------------------------------
# 退出码：拒绝、未实现、通过要能被脚本区分开
# ---------------------------------------------------------------------------


def test_template_is_itself_valid(tmp_path, template, capsys):
    """模板必须能直接通过验证——否则它就是一份误导人的样例。"""
    assert runrbfe.main(["validate", "--config", _write(tmp_path, "e.json", template)]) == (
        runrbfe.EXIT_OK
    )


def test_rejection_uses_a_distinct_exit_code(tmp_path, template):
    template["ligand_B"]["formal_charge"] = -1
    code = runrbfe.main(["validate", "--config", _write(tmp_path, "bad.json", template)])
    assert code == runrbfe.EXIT_REJECTED
    assert runrbfe.EXIT_REJECTED != runrbfe.EXIT_OK


@pytest.mark.parametrize("cmd", ["prepare", "run", "analyze"])
def test_unimplemented_subcommands_exit_with_their_own_code(cmd, capsys):
    """「尚未实现」必须和「输入被拒绝」区分开——两者的后续动作完全不同。"""
    assert runrbfe.main([cmd]) == runrbfe.EXIT_NOT_IMPLEMENTED
    assert runrbfe.EXIT_NOT_IMPLEMENTED not in (runrbfe.EXIT_OK, runrbfe.EXIT_REJECTED)
    assert "尚未实现" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 配置加载 fail-closed
# ---------------------------------------------------------------------------


def test_unknown_field_is_rejected_not_ignored(tmp_path, template):
    """🔑 `temperature_kelvim` 写错一个字母，静默忽略就会用默认温度跑完全程。"""
    template["protocol"]["temperature_kelvim"] = 310.0
    code = runrbfe.main(["validate", "--config", _write(tmp_path, "typo.json", template)])
    assert code == runrbfe.EXIT_REJECTED


def test_unknown_field_at_every_level(tmp_path, template):
    for mutate in (
        lambda d: d.update(bogus=1),
        lambda d: d["environment"].update(bogus=1),
        lambda d: d["ligand_A"].update(bogus=1),
    ):
        payload = copy.deepcopy(template)
        mutate(payload)
        assert runrbfe.main(
            ["validate", "--config", _write(tmp_path, "x.json", payload)]
        ) == runrbfe.EXIT_REJECTED


def test_missing_required_field_is_rejected(tmp_path, template):
    del template["protocol"]["seed"]
    assert runrbfe.main(
        ["validate", "--config", _write(tmp_path, "m.json", template)]
    ) == runrbfe.EXIT_REJECTED


def test_malformed_json_is_rejected(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    assert runrbfe.main(["validate", "--config", str(path)]) == runrbfe.EXIT_REJECTED


def test_non_object_top_level_is_rejected(tmp_path):
    assert runrbfe.main(
        ["validate", "--config", _write(tmp_path, "list.json", [1, 2, 3])]
    ) == runrbfe.EXIT_REJECTED


# ---------------------------------------------------------------------------
# 输出必须诚实
# ---------------------------------------------------------------------------


def test_pass_output_says_what_was_not_checked(tmp_path, template, capsys):
    """PASS 不等于「全都查过了」——环变化/手性/共价需要 R1 的原子映射。"""
    runrbfe.main(["validate", "--config", _write(tmp_path, "e.json", template)])
    out = capsys.readouterr().out
    assert "未检查" in out
    assert "通过不等于全都查过了" in out


def test_rejection_message_preserves_scope_vs_capability_distinction(
    tmp_path, template, capsys
):
    """计划 §2：这些是本项目首版的范围限制，不代表 RBFE 方法普遍不支持。"""
    template["environment"]["is_membrane"] = True
    runrbfe.main(["validate", "--config", _write(tmp_path, "mem.json", template)])
    err = capsys.readouterr().err
    assert "不代表 RBFE 方法普遍不支持" in err


def test_json_report_is_machine_readable(tmp_path, template, capsys):
    runrbfe.main(["validate", "--config", _write(tmp_path, "e.json", template), "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{") :])
    assert payload["ok"] is True
    assert payload["manifest"]["direction"] == rc.RBFE_DIRECTION
    assert payload["unchecked"]


# ---------------------------------------------------------------------------
# combine：符号与误差传播（R0 验收的第二条）
# ---------------------------------------------------------------------------


def _leg_payload(phase, dg, se):
    return {
        "phase": phase,
        "edge_id": "E1",
        "ligand_A": "A",
        "ligand_B": "B",
        "delta_g_A_to_B": dg,
        "stderr": se,
        "energy_unit": rc.KJ_PER_MOL,
        "uncertainty_method": "mbar",
        "n_effective_samples": 500,
        "quality_gate_passed": True,
        "artifacts_fingerprint": "abc",
    }


def test_combine_reports_correct_sign_and_error(tmp_path, capsys):
    c = _write(tmp_path, "c.json", _leg_payload("complex", -10.0, 3.0))
    s = _write(tmp_path, "s.json", _leg_payload("solvent", -2.0, 4.0))
    assert runrbfe.main(["combine", "--complex-json", c, "--solvent-json", s]) == runrbfe.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["ddG_bind_B_minus_A"] == pytest.approx(-8.0)
    assert payload["ddG_stderr"] == pytest.approx(5.0)
    assert payload["direction"] == rc.RBFE_DIRECTION


def test_combine_accepts_covariance(tmp_path, capsys):
    c = _write(tmp_path, "c.json", _leg_payload("complex", -10.0, 3.0))
    s = _write(tmp_path, "s.json", _leg_payload("solvent", -2.0, 4.0))
    runrbfe.main(
        ["combine", "--complex-json", c, "--solvent-json", s, "--covariance", "6.0"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ddG_stderr"] == pytest.approx(13.0**0.5)


def test_combine_rejects_swapped_legs(tmp_path):
    """把 solvent 结果当 complex 传进来必须报错，而不是算出一个符号翻转的数。"""
    c = _write(tmp_path, "c.json", _leg_payload("solvent", -10.0, 3.0))
    s = _write(tmp_path, "s.json", _leg_payload("solvent", -2.0, 4.0))
    assert runrbfe.main(
        ["combine", "--complex-json", c, "--solvent-json", s]
    ) == runrbfe.EXIT_REJECTED


def test_combine_default_covariance_is_zero_and_documented():
    """默认独立是个**假设**，不是事实；help 里必须说清楚代价。"""
    parser = runrbfe.build_parser()
    help_text = parser.format_help()
    assert "combine" in help_text
    combine_help = [
        a for a in parser._subparsers._group_actions[0].choices["combine"]._actions
        if a.dest == "covariance"
    ][0]
    assert combine_help.default == 0.0
    assert "误差被低估" in combine_help.help


# ---------------------------------------------------------------------------
# 职责边界
# ---------------------------------------------------------------------------


def test_cli_holds_no_science():
    """§4：runrbfe.py 不持有科学算法，也不持有第二份计算流程。

    它不得自己算 ΔΔG——那只能来自 rbfe_core.combine_rbfe。
    """
    from pathlib import Path

    source = Path(runrbfe.__file__).read_text(encoding="utf-8")
    assert "combine_rbfe" in source
    # 不得出现自建的两腿相减
    assert "complex_leg.delta_g -" not in source
    assert "delta_g - " not in source


def test_cli_does_not_import_openmm():
    from pathlib import Path

    source = Path(runrbfe.__file__).read_text(encoding="utf-8")
    assert "import openmm" not in source

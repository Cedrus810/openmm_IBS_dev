"""[CLI-01, 2026-09-02] 用户诊断命令 + 惰性 import 的回归测试。

两组契约：

1. `abfe_diagnostics` 的 `validate-config` / `doctor` 必须是只读诊断，且
   **不得**变成启动硬门：`RunConfig` 遇到未知键仍然照常合并（2026-08-24 那次
   启动期硬拒绝未知键炸掉 resume 的事故不许重演）。
2. `abfe_core` / `ibs_engine` 里 torch / openmmml / pymbar 必须是惰性 import：
   只 import 这两个模块不许把它们拖进来，但 `HAS_ORB` / `HAS_PYMBAR` 这两个
   模块属性仍然要能读到布尔值。
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import abfe_diagnostics  # noqa: E402


# ---------------------------------------------------------------------------
# 惰性 import
# ---------------------------------------------------------------------------
_LAZY_PROBE = """
import sys
import {module}
loaded = [m for m in ("torch", "openmmml", "pymbar") if m in sys.modules]
print(",".join(loaded))
"""


@pytest.mark.parametrize("module", ["abfe_core", "ibs_engine"])
def test_importing_core_modules_does_not_pull_torch_or_pymbar(module):
    """import abfe_core / ibs_engine 不得连带 import torch / openmmml / pymbar。

    这三个是 `runabfe.py --help` 4.3 s 里最大的两块（torch 1.31 s、
    pymbar 1.12 s，`python -X importtime` 实测）。它们在子进程里探测，
    因为本进程早就被别的测试 import 过了。
    """
    proc = subprocess.run(
        [sys.executable, "-c", _LAZY_PROBE.format(module=module)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    leaked = [name for name in proc.stdout.strip().split(",") if name]
    assert leaked == [], f"import {module} 把这些模块拖进来了: {leaked}"


@pytest.mark.parametrize(
    "module_name,attr",
    [
        ("abfe_core", "HAS_PYMBAR"),
        ("abfe_core", "HAS_ORB"),
        ("ibs_engine", "HAS_PYMBAR"),
    ],
)
def test_capability_flags_still_readable_as_module_attributes(module_name, attr):
    """外部读 `abfe_core.HAS_PYMBAR` 这种写法必须继续拿到布尔值。"""
    module = __import__(module_name)
    assert isinstance(getattr(module, attr), bool)


def test_unknown_module_attribute_still_raises_attribute_error():
    """补上的 `__getattr__` 不能把任意名字都变成合法属性。"""
    import abfe_core

    with pytest.raises(AttributeError):
        abfe_core.definitely_not_a_real_symbol


def test_core_modules_have_no_bare_capability_flag_reads():
    """模块内部不许再出现裸 `HAS_PYMBAR` / `HAS_ORB` 读取。

    模块级 `__getattr__` 只对**外部**属性访问生效；模块内部的全局名查找不走
    它，写成裸名字会直接 NameError。这条测试把这个陷阱钉住。
    """
    offenders = []
    for filename in ("abfe_core.py", "ibs_engine.py"):
        path = os.path.join(REPO_ROOT, filename)
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in ("HAS_PYMBAR", "HAS_ORB")
            ):
                offenders.append(f"{filename}:{node.lineno}")
    assert offenders == [], (
        "这些地方裸读了 HAS_PYMBAR/HAS_ORB，会 NameError；改成 has_pymbar()/"
        f"has_orb(): {offenders}"
    )


# ---------------------------------------------------------------------------
# 配置键提取
# ---------------------------------------------------------------------------
def test_consumed_config_keys_cover_the_real_preset_assignments():
    """静态提取出的键集合必须覆盖 `RunConfig.__init__` 里所有 `preset[...] = `。

    这是 `validate-config` 判"未知键"的依据。漏了会把合法键误报成未知键。
    """
    tree = ast.parse(open(os.path.join(REPO_ROOT, "runabfe.py"), encoding="utf-8").read())
    run_config = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RunConfig"
    )
    init = next(
        n for n in run_config.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    assigned = {
        node.slice.value
        for node in ast.walk(init)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "preset"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    assert assigned, "没能从 RunConfig.__init__ 里提到任何 preset 键，提取逻辑坏了"
    consumed = abfe_diagnostics.collect_consumed_config_keys()
    missing = sorted(assigned - set(consumed))
    assert missing == [], f"这些真实配置键没被 collect_consumed_config_keys 认出: {missing}"


def test_temperature_typo_is_reported_with_a_suggestion(tmp_path, capsys):
    """`temprature` 这种拼写错误必须被指出来，并给出 `temperature` 的建议。"""
    config = tmp_path / "typo.json"
    config.write_text(json.dumps({"temprature": 310}), encoding="utf-8")
    exit_code = abfe_diagnostics.validate_config_main(
        ["--config", str(config), "--json"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    typo = [f for f in payload["findings"] if "temprature" in f["message"]]
    assert typo, "拼错的 temprature 没有被报出来"
    assert typo[0]["level"] == "ERROR"
    assert "temperature" in typo[0]["detail"]


def test_valid_config_reports_no_errors(tmp_path, capsys):
    """一份键名和取值都正确的配置不应该报错。"""
    gro = tmp_path / "system.gro"
    gro.write_text("placeholder\n", encoding="utf-8")
    config = tmp_path / "ok.json"
    config.write_text(
        json.dumps(
            {
                "_comment": "注释键应当被忽略",
                "mode": "ibs",
                "decoupling": "dual_lambda",
                "potential": "softcore",
                "temperature": 300.0,
                "n_steps_per_window": 250000,
                "gro": str(gro),
            }
        ),
        encoding="utf-8",
    )
    exit_code = abfe_diagnostics.validate_config_main(
        ["--config", str(config), "--json"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0, [f for f in payload["findings"] if f["level"] == "ERROR"]
    assert payload["n_errors"] == 0


def test_bad_choice_value_is_an_error(tmp_path, capsys):
    """取值不在 argparse choices 里必须报错（choices 从真 parser 上读）。"""
    config = tmp_path / "bad.json"
    config.write_text(json.dumps({"mode": "not_a_mode"}), encoding="utf-8")
    exit_code = abfe_diagnostics.validate_config_main(
        ["--config", str(config), "--json"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert any(
        f["level"] == "ERROR" and "mode" in f["message"] for f in payload["findings"]
    )


def test_cli_only_key_is_not_reported_as_a_typo(tmp_path, capsys):
    """`preset` 是命令行选项名而不是配置键——要说清楚，不要给"你是不是想写 reset"。"""
    config = tmp_path / "cli_only.json"
    config.write_text(json.dumps({"preset": "production"}), encoding="utf-8")
    abfe_diagnostics.validate_config_main(["--config", str(config), "--json"])
    payload = json.loads(capsys.readouterr().out)
    finding = next(f for f in payload["findings"] if "`preset`" in f["message"])
    assert finding["level"] == "WARN"
    assert "--preset" in finding["message"]
    # 不该走"最接近的已知键"那条拼写建议分支（`--preset` 本身就含 "reset"，
    # 所以这里判的是提示语本身，不是子串）。
    assert "最接近的已知键" not in finding["detail"]


def test_missing_input_path_is_an_error(tmp_path, capsys):
    config = tmp_path / "paths.json"
    config.write_text(
        json.dumps({"gro": str(tmp_path / "nope.gro")}), encoding="utf-8"
    )
    exit_code = abfe_diagnostics.validate_config_main(
        ["--config", str(config), "--json"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert any("gro" in f["message"] and f["level"] == "ERROR" for f in payload["findings"])


# ---------------------------------------------------------------------------
# 不得成为启动硬门
# ---------------------------------------------------------------------------
def test_run_config_still_accepts_unknown_keys(tmp_path, monkeypatch):
    """未知键只在 `validate-config` 里报——`RunConfig` 必须照常合并。

    2026-08-24 的事故：启动期硬拒绝未知键直接炸掉 resume。这条测试防止
    诊断功能被"顺手"接进启动路径。
    """
    import runabfe

    config = tmp_path / "unknown.json"
    config.write_text(
        json.dumps({"temprature": 310, "temperature": 300.0}), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys, "argv", ["runabfe.py", "--config", str(config), "--ligand", "MOL"]
    )
    cfg = runabfe.RunConfig(runabfe.parse_arguments())
    assert cfg.data["temprature"] == 310, "未知键被丢掉了（原行为是保留）"
    assert cfg.temperature == 300.0, "有效 temperature 不应被拼错的键影响"


# ---------------------------------------------------------------------------
# 子命令分流
# ---------------------------------------------------------------------------
def test_diagnostic_commands_are_dispatched_before_the_main_parser():
    """诊断子命令必须在主 parser 之前分流。

    `--json` 这类只属于诊断子命令的选项，主 parser 不认识；交给它解析会
    直接 SystemExit(2)。
    """
    import runabfe

    assert runabfe.dispatch_diagnostic_command([]) is None
    assert runabfe.dispatch_diagnostic_command(["--resume"]) is None
    assert set(runabfe.DIAGNOSTIC_COMMANDS) == {
        "doctor",
        "validate-config",
        "config-template",
    }


def test_diagnostic_commands_are_documented_in_help():
    """两个命令要出现在 `--help` 里，否则用户找不到。"""
    import runabfe

    help_text = runabfe.build_parser().format_help()
    assert "validate-config" in help_text
    assert "doctor" in help_text
    assert "config-template" in help_text


def test_build_parser_does_not_consume_argv(monkeypatch):
    """`build_parser()` 只构造 parser，不解析 argv。"""
    import runabfe

    monkeypatch.setattr(sys, "argv", ["runabfe.py", "--this-flag-does-not-exist"])
    parser = runabfe.build_parser()
    assert parser.prog is not None


def _unresolved_loads(path: str, names: set) -> list:
    """返回 `path` 里所有"读了 `names` 里的名字、但该名字在本作用域链上从未
    定义/导入"的位置。

    这是惰性化 import 最容易留下的坑：把模块级 `import pymbar` 删掉之后，
    某个函数体里还留着一处裸 `pymbar.xxx`——它只在那条分支被走到时才
    NameError，普通测试常常覆盖不到。改这几个模块时这条测试是最后一道闸。
    """
    tree = ast.parse(open(path, encoding="utf-8").read())
    hits = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.scopes = [set()]

        def visit_FunctionDef(self, node):
            local = {a.arg for a in node.args.args + node.args.kwonlyargs}
            if node.args.vararg:
                local.add(node.args.vararg.arg)
            if node.args.kwarg:
                local.add(node.args.kwarg.arg)
            for inner in ast.walk(node):
                if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Store):
                    local.add(inner.id)
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    for alias in inner.names:
                        local.add((alias.asname or alias.name).split(".")[0])
            self.scopes.append(local)
            self.generic_visit(node)
            self.scopes.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Name(self, node):
            if (
                isinstance(node.ctx, ast.Load)
                and node.id in names
                and not any(node.id in scope for scope in self.scopes)
            ):
                hits.append(f"{os.path.basename(path)}:{node.lineno} → {node.id}")

    Visitor().visit(tree)
    return hits


@pytest.mark.parametrize(
    "filename", ["abfe_core.py", "ibs_engine.py", "runabfe.py", "abfe_diagnostics.py"]
)
def test_no_unresolved_lazy_module_names(filename):
    """惰性化之后不许再有裸 `torch` / `pymbar` / `MLPotential` 读取。

    改用 `_require_torch()` / `_require_pymbar()` / `_require_mlpotential()`，
    或在函数内显式 import。
    """
    offenders = _unresolved_loads(
        os.path.join(REPO_ROOT, filename), {"torch", "pymbar", "MLPotential"}
    )
    assert offenders == [], (
        "这些位置会 NameError（模块级 import 已被惰性化）："
        f"{offenders}"
    )


# ---------------------------------------------------------------------------
# config-template
# ---------------------------------------------------------------------------
def _template(argv, capsys):
    exit_code = abfe_diagnostics.config_template_main(argv)
    out = capsys.readouterr().out
    return exit_code, json.loads(out)


def test_template_round_trips_through_validate_config(capsys):
    """生成的模板必须能被 `validate-config` 零错误接受。

    这是"模板→改→自查"这个闭环唯一的硬要求：模板自己就报错的话，用户根本
    分不清是自己填错了还是模板本来就不对。
    """
    exit_code, payload = _template([], capsys)
    assert exit_code == 0
    template_keys = [k for k in payload if not k.startswith("_")]
    assert template_keys, "模板一个真实键都没有"

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tpl.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        check_code = abfe_diagnostics.validate_config_main(
            ["--config", path, "--json"]
        )
        report = json.loads(capsys.readouterr().out)
    assert check_code == 0, [f for f in report["findings"] if f["level"] == "ERROR"]
    assert report["n_errors"] == 0


@pytest.mark.parametrize("argv", [[], ["--all"]])
def test_template_never_emits_machine_local_or_internal_keys(argv, capsys):
    """模板不许带机器本地路径的值，也不许列运行期回填的 provenance 字段。

    `gmx_path` 在仓库配置里是 `/home/ruigengji/gmx26.0C`——抄进模板发给别人就是
    错的；`repeat_seed_source` 这类由代码回填的键被用户预先写死会让快照说谎。
    """
    _, payload = _template(argv, capsys)
    for key in abfe_diagnostics._MACHINE_LOCAL_KEYS + abfe_diagnostics._REQUIRED_INPUT_KEYS:
        if key in payload:
            assert payload[key] is None, f"{key} 必须留空，实际 {payload[key]!r}"
    leaked = sorted(k for k in payload if k in abfe_diagnostics._NON_USER_KEYS)
    assert leaked == [], f"模板泄漏了非用户键: {leaked}"


def test_template_prefers_production_values_over_argparse_none_sentinels(capsys):
    """argparse 默认是 `None`（"没给 flag"哨兵）或与生产相反时，取生产配置的值。

    反例就是 `enable_gradual_warmup`：`store_true` ⟹ argparse 默认 `False`，
    而仓库生产配置是 `true`。照抄 argparse 默认会生成一份跑不出生产行为的模板。
    """
    _, payload = _template([], capsys)
    shipped = json.load(
        open(os.path.join(REPO_ROOT, "abfe_config.json"), encoding="utf-8")
    )
    checked = 0
    for key, shipped_value in shipped.items():
        if key.startswith("_") or key not in payload:
            continue
        if key in abfe_diagnostics._MACHINE_LOCAL_KEYS:
            continue
        if key in abfe_diagnostics._REQUIRED_INPUT_KEYS:
            continue
        import runabfe

        if key in runabfe.PRESET_CONFIGS["production"]:
            continue  # 预设优先，本例不比
        assert payload[key] == shipped_value, (
            f"{key}: 模板给 {payload[key]!r}，生产配置是 {shipped_value!r}"
        )
        checked += 1
    assert checked > 10, f"只比到 {checked} 个键，取值优先级没被真正覆盖到"


def test_template_drops_cli_only_keys(capsys):
    """`preset` 只能从命令行给，模板不该教人写进配置文件。"""
    _, payload = _template([], capsys)
    assert "preset" not in payload


def test_template_all_is_a_superset(capsys):
    """`--all` 只增不减。"""
    _, base = _template([], capsys)
    _, full = _template(["--all"], capsys)
    base_keys = {k for k in base if not k.startswith("_")}
    full_keys = {k for k in full if not k.startswith("_")}
    assert base_keys < full_keys


def test_template_rejects_unknown_preset(capsys):
    assert abfe_diagnostics.config_template_main(["--preset", "no_such_preset"]) == 1


def test_template_out_writes_a_file(tmp_path, capsys):
    path = tmp_path / "written.json"
    assert abfe_diagnostics.config_template_main(["--out", str(path)]) == 0
    capsys.readouterr()
    assert json.loads(path.read_text(encoding="utf-8"))

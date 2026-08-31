"""P1-04 / P1-15：analyze-only 的协议身份校验 + refine 窗口编号解析。

## P1-04 缺陷是什么（已修，本文件现在是契约测试）

1. `main()` 在加载 DEXP 参数、要求 `--ligand` 等**模拟输入**之后才分流
   analyze-only——纯分析现有结果被无关前置校验阻断；
2. `run_post_analysis` 读 run_provenance.json 但 temperature/mode/decoupling
   仍取本次命令/预设默认值——原运行是 310 K 时会用 300 K 的 kT/Boresch 修正；
3. 回退分析的两个分支 fail-open：阶段 checkpoint 不检查 converged/协议身份/
   覆盖证据；原始窗口分支只检查"现有编号连续"，16 窗丢末 3 窗后剩下的
   0..12 依然连续，缺末窗拼出的截断 ΔG 照样放行。

## P1-15 缺陷是什么（已修）

`abfe_preoptimizer.refine_stage_lambda_path_from_data` 用 `sorted(glob)`
的字典序 + enumerate 位置当窗口编号——≥10 窗时 window_10 排在 window_2
之前，u_kn/bias/base 与 window_ranges 错配，写出错误的新 λ 路径。

## 修法（2026-08-30）

- main 最先分流 analyze-only；
- run_post_analysis 默认从 run_provenance.json 恢复 temperature/mode/
  decoupling，仅显式 CLI 覆盖并留审计记录；
- checkpoint 分支要求 converged is True + protocol_key + 覆盖/端点证据；
- 原始窗口分支要求文件数 == preopt window_ranges 数，并用生产同款
  `_assert_expected_windows_all_loaded` 检查 λ 状态覆盖；
- refine 从文件名解析整数编号、按数值排序，重复/缺失一律拒绝。

## 不要这样让本文件变绿

把 analyze-only 分流挪回去、让温度覆盖不留审计、把 converged 缺席当放行、
或让 refine 恢复字典序排序。
"""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

import runabfe
from runabfe import _assert_traditional_protocol_supported  # noqa: F401  (导入健康检查)
import abfe_preoptimizer as preopt

ROOT = Path(__file__).resolve().parents[1]
RUNABFE_PATH = ROOT / "runabfe.py"
PREOPT_PATH = ROOT / "abfe_preoptimizer.py"


# ---------------------------------------------------------------------------
# 1. P1-04a：analyze-only 最先分流
# ---------------------------------------------------------------------------


def test_analyze_only_is_dispatched_before_dexp_and_ligand_checks():
    src = RUNABFE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    main_fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    dispatch_line = None
    dexp_line = None
    ligand_line = None
    for node in ast.walk(main_fn):
        if isinstance(node, ast.If) and "analyze_only" in ast.unparse(node.test):
            if dispatch_line is None:
                dispatch_line = node.lineno
        if isinstance(node, ast.Assign):
            target_src = ast.unparse(node)
            if "dexp_params = _load_dexp_params_fail_closed(" in target_src:
                dexp_line = node.lineno
        if isinstance(node, ast.If):
            test_src = ast.unparse(node.test)
            if "not config.ligand" in test_src and ligand_line is None:
                ligand_line = node.lineno
    assert dispatch_line is not None, "main 必须分流 analyze_only"
    assert dexp_line is not None and ligand_line is not None
    assert dispatch_line < dexp_line and dispatch_line < ligand_line, (
        "analyze-only 分流必须先于 DEXP 参数加载与 ligand 要求（P1-04）"
    )


# ---------------------------------------------------------------------------
# 2. P1-04b：provenance 恢复（行为测试）
# ---------------------------------------------------------------------------


class _ArgsStub:
    def __init__(self, **kwargs):
        self.output = str(kwargs["output"])
        self.temperature = kwargs.get("temperature", 300.0)
        self.mode = kwargs.get("mode", "ibs")
        self.decoupling = kwargs.get("decoupling", "dual_lambda")


def _make_output_dir(tmp_path, provenance_cfg):
    out = tmp_path / "out"
    out.mkdir()
    (out / "run_provenance.json").write_text(
        json.dumps({"config": provenance_cfg}), encoding="utf-8"
    )
    return out


def _run_recovery(tmp_path, provenance_cfg, argv, **arg_kwargs):
    """直接执行 run_post_analysis 开头的恢复逻辑：让它在 Boresch 参数查找处
    抛 FileNotFoundError（没有 traditional_complex 目录），我们只关心恢复后
    的 args 字段——通过给 temp 换算留痕来观察。
    """
    out = _make_output_dir(tmp_path, provenance_cfg)
    args = _ArgsStub(output=out, **arg_kwargs)
    old_argv = sys.argv
    sys.argv = argv
    try:
        try:
            runabfe.run_post_analysis(args)
        except FileNotFoundError:
            pass
        except Exception:
            pass
    finally:
        sys.argv = old_argv
    return args


def test_temperature_recovered_from_provenance(tmp_path, monkeypatch):
    args = _run_recovery(
        tmp_path,
        {"temperature": 310.0, "mode": "ibs", "decoupling": "dual_lambda"},
        ["runabfe.py"],
        temperature=300.0,
    )
    assert float(args.temperature) == pytest.approx(310.0), (
        "原运行 310 K、本次未显式覆盖 ⟹ analyze-only 必须恢复 310 K（P1-04）"
    )


def test_explicit_cli_override_wins(tmp_path):
    args = _run_recovery(
        tmp_path,
        {"temperature": 310.0, "mode": "ibs", "decoupling": "dual_lambda"},
        ["runabfe.py", "--temperature", "298.0"],
        temperature=298.0,
    )
    assert float(args.temperature) == pytest.approx(298.0)


def test_missing_provenance_keeps_cli_values(tmp_path):
    out = tmp_path / "out_noprov"
    out.mkdir()
    args = _ArgsStub(output=out, temperature=300.0)
    old_argv = sys.argv
    sys.argv = ["runabfe.py"]
    try:
        try:
            runabfe.run_post_analysis(args)
        except Exception:
            pass
    finally:
        sys.argv = old_argv
    assert float(args.temperature) == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# 3. P1-04c：回退分析两个分支的静态守护
# ---------------------------------------------------------------------------


def test_checkpoint_branch_requires_converged_protocol_and_coverage():
    import inspect
    from abfe_pipeline import ABFEPipeline
    src = inspect.getsource(runabfe._analyze_dual_leg_artifacts)
    assert "_validate_stage_checkpoint(" in src
    shared = inspect.getsource(ABFEPipeline._validate_stage_checkpoint)
    assert "_assert_reusable_stage_cache_sane(" in shared
    assert "expected_protocol_key" in shared and "lambda_path_fingerprint" in shared


def test_fallback_branch_checks_expected_window_count_and_uses_shared_helper():
    import inspect
    import ibs_engine
    src = inspect.getsource(runabfe._analyze_dual_leg_artifacts)
    assert "load_ibs_window_outputs_from_dir(" in src
    shared = inspect.getsource(ibs_engine.load_ibs_window_outputs_from_dir)
    assert "_assert_expected_windows_all_loaded(" in shared
    assert "_load_validated_window_data_triplet(" in shared
    assert "production_entry_f_k" in shared


# ---------------------------------------------------------------------------
# 4. P1-15：refine 按数值编号映射窗口（行为测试）
# ---------------------------------------------------------------------------


def _write_window_files(stage_dir, window_ranges, *, two_digit_names):
    """给每个窗口写可区分的 u_kn/bias/base：u_kn 内容编码真实窗口编号。"""
    stage_dir.mkdir(parents=True, exist_ok=True)
    for w_idx, (start, end) in enumerate(window_ranges):
        n_states = end - start
        # 用窗口编号本身填充 u_kn —— 任何错位都会在映射检查里现形。
        u_kn = np.full((n_states, 4), float(w_idx), dtype=np.float64)
        bias = np.zeros((4,), dtype=np.float64)
        base = np.zeros((4,), dtype=np.float64)
        e_path = stage_dir / f"dual_window_{w_idx}_vdw_energies.npy"
        np.save(e_path, u_kn)
        np.save(stage_dir / f"dual_window_{w_idx}_vdw_bias.npy", bias)
        np.save(stage_dir / f"dual_window_{w_idx}_vdw_base.npy", base)
        assert e_path.exists()


def test_refine_maps_two_digit_windows_numerically(tmp_path, monkeypatch):
    """12 个窗口（0..11）必须按数值映射；字典序会立刻把 10/11 排到 2 前面。"""
    window_ranges = [(i, i + 1) for i in range(12)]
    stage_dir = tmp_path / "decharging"
    _write_window_files(stage_dir, window_ranges, two_digit_names=True)
    preopt_file = tmp_path / "preopt_dual_vanishing.json"
    lambdas_var = np.linspace(1.0, 0.0, 12).tolist()
    preopt_file.write_text(
        json.dumps({"lambdas_var": lambdas_var, "window_ranges": window_ranges}),
        encoding="utf-8",
    )

    captured = {}

    def _fake_solve(window_data, kt, stage_name=None):
        captured["indices"] = [d["lambda_indices"] for d in window_data]
        captured["first_col"] = [float(d["u_kn"][0, 0]) for d in window_data]
        raise RuntimeError("stop-before-solve")  # 只验证加载映射，不需要真求解

    monkeypatch.setattr(
        "ibs_engine.solve_stage_integrated", _fake_solve
    )
    with pytest.raises(RuntimeError, match="stop-before-solve"):
        preopt.refine_stage_lambda_path_from_data(
            stage_dir=str(stage_dir),
            preopt_path=str(preopt_file),
            temperature_k=300.0,
            stage_type="vdw",
        )
    # 位置 i 必须加载编号 i 的文件：u_kn 首元素 == 窗口编号。
    for position, value in enumerate(captured["first_col"]):
        assert value == float(position), (
            f"位置 {position} 加载的是窗口 {int(value)} 的数据 —— 字典序错配（P1-15）"
        )
    assert captured["indices"] == [[i] for i in range(12)]


def test_refine_rejects_missing_window_index(tmp_path, monkeypatch):
    """缺一个编号（0..2,4..11，共 11 个文件）⟹ 拒绝，不得静默错配。"""
    window_ranges = [(i, i + 1) for i in range(12)]
    stage_dir = tmp_path / "decharging"
    for w_idx, (_s, _e) in enumerate(window_ranges):
        if w_idx == 3:
            continue  # 缺窗口 3
        n_states = 1
        u_kn = np.full((n_states, 4), float(w_idx), dtype=np.float64)
        stage_dir.mkdir(parents=True, exist_ok=True)
        np.save(stage_dir / f"dual_window_{w_idx}_vdw_energies.npy", u_kn)
        np.save(stage_dir / f"dual_window_{w_idx}_vdw_bias.npy", np.zeros(4))
        np.save(stage_dir / f"dual_window_{w_idx}_vdw_base.npy", np.zeros(4))
    preopt_file = tmp_path / "preopt_dual_vanishing.json"
    preopt_file.write_text(
        json.dumps(
            {
                "lambdas_var": np.linspace(1.0, 0.0, 12).tolist(),
                "window_ranges": window_ranges,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="不一致|连续编号"):
        preopt.refine_stage_lambda_path_from_data(
            stage_dir=str(stage_dir),
            preopt_path=str(preopt_file),
            temperature_k=300.0,
            stage_type="vdw",
        )


# ---------------------------------------------------------------------------
# 小工具：ast 辅助（独立命名，避免与本文件其它定义混淆）
# ---------------------------------------------------------------------------


def ast_parse(src: str):
    import ast

    return ast.parse(src)


def ast_walk(node):
    import ast

    return ast.walk(node)


ast_FunctionDef = None


def ast_FunctionDef():  # noqa: F811 - 保持属性式引用可用
    import ast

    return ast.FunctionDef

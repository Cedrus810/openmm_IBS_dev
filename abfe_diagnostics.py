"""只读用户诊断：`runabfe.py doctor` / `validate-config` / `config-template`。

[CLI-01, 2026-09-02] 这两个命令是发布准备度评估里"用户可读的配置与运行诊断"
那一条（见 `docs/RELEASE_READINESS_2026-08-31.md`《发布前应补齐的交付项》）。

设计约束，改这个文件前先读：

1. **只读。** 不建 OpenMM Context、不积分、不写任何运行产物、不碰缓存和
   checkpoint。`doctor` 会 import openmm 去列平台（那是它的职责），但不建 Context。
2. **不是启动硬门。** `runabfe.py` 的正常启动路径**不得**因为这里报错而拒绝运行。
   2026-08-24 有一次在启动期加 unknown-key 硬校验、直接炸掉 resume 的事故
   （见 `runabfe.py` `RunConfig.__init__` 里那段注释），结论是：拼写检查只允许
   出现在这种显式调用的诊断命令里，运行期最多告警。
3. **不复制参数表。** 校验用的 dest/type/choices/default 全部从
   `runabfe.build_parser()` 这个真 parser 上读，配置键从 `RunConfig.__init__`
   的源码里静态提取。两者都不在这里维护第二份副本——副本必然和本体各改一半。
"""

from __future__ import annotations

import argparse
import ast
import difflib
import importlib
import json
import os
import shutil
import subprocess
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# 诊断结论的三档。ERROR 会让命令以非零退出码结束；WARN 不会。
LEVEL_OK = "OK"
LEVEL_WARN = "WARN"
LEVEL_ERROR = "ERROR"

_LEVEL_MARK = {LEVEL_OK: "[OK]", LEVEL_WARN: "[WARN] ", LEVEL_ERROR: "[ERR]"}

# 这些配置键的值是路径，`validate-config` 会检查存在性。
# 只列**输入**路径：输出目录不在这里（它本来就允许不存在，由运行时创建）。
_PATH_LIKE_INPUT_KEYS = (
    "gro",
    "top",
    "ligand_xml",
    "dexp_params",
    "torsion_params",
    "boresch_anchors",
    "boresch_orb",
    "membrane_input_declaration",
    "co_alchemical_ion",
    "apbs_evidence",
    "charging_rerun_dir",
    "attachment_rerun_dir",
    "force_switch_deviation_evidence",
)

# 这些键会被下游模块（abfe_pipeline / ibs_engine / abfe_preoptimizer）从
# 配置里读走，但不是 `runabfe.py` 里的 `preset[...] = args.*` 覆盖项，所以
# 静态提取抓不到。加到这里 = 声明"它确实会被消费"。
#
# 加新键时的判据：代码里存在一处从**运行配置**读它的 `config.get("<key>")` /
# `config.data["<key>"]`，而不是从别的字典读同名字段。
_EXTRA_CONSUMED_KEYS = (
    "repeat_seed",
    "enable_lambda_refine",
    "refine_max_window_span_kJ",
    "refine_n_steps_per_window",
    "refine_steps_per_update",
    "n_states_per_stage",
    "allow_untrusted_stage_results",
    "enable_equilibration_convergence_stop",
    "attachment_seed",
    "attachment_equil_steps_per_state",
    "attachment_steps_per_sample",
    "pilot_n_steps_per_state",
    "pilot_finite_difference_delta",
    "pilot_shadow_checkpoint_interval",
    "max_bias_updates",
    "min_bias_updates",
    "max_bias_warmup_steps",
    "required_consecutive_bias_updates",
    "ibs_lse_log_residual_tolerance",
    "timestep_ps",
)


def _display_width(text: str) -> int:
    """字符串在等宽终端里的显示宽度（CJK 全角字符算 2 列）。

    报告里的键名是中英混排，`str.ljust` 按**字符数**补空格会让列对不齐。
    """
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _display_width(text))


# `--platform` 没有 argparse choices（见 validate_config_main 里的说明），
# 这里只是常识提示用的名单，不是权威列表。
_WELL_KNOWN_OPENMM_PLATFORMS = ("CUDA", "OpenCL", "CPU", "Reference", "HIP")


class Finding:
    """一条诊断结论。"""

    __slots__ = ("level", "section", "message", "detail")

    def __init__(self, level: str, section: str, message: str, detail: str = ""):
        self.level = level
        self.section = section
        self.message = message
        self.detail = detail

    def as_dict(self) -> Dict[str, str]:
        return {
            "level": self.level,
            "section": self.section,
            "message": self.message,
            "detail": self.detail,
        }


class Report:
    """诊断结果收集器。渲染成人读文本或 JSON。"""

    def __init__(self, title: str, section_order: Tuple[str, ...] = ()):
        self.title = title
        self.section_order = section_order
        self.findings: List[Finding] = []
        self.facts: List[Tuple[str, str, str]] = []  # (section, name, value)

    def add(self, level: str, section: str, message: str, detail: str = "") -> None:
        self.findings.append(Finding(level, section, message, detail))

    def fact(self, section: str, name: str, value: Any) -> None:
        self.facts.append((section, name, "" if value is None else str(value)))

    @property
    def n_errors(self) -> int:
        return sum(1 for f in self.findings if f.level == LEVEL_ERROR)

    @property
    def n_warnings(self) -> int:
        return sum(1 for f in self.findings if f.level == LEVEL_WARN)

    def render_text(self) -> str:
        out: List[str] = [self.title, "=" * len(self.title), ""]
        # 小节按 `section_order` 排；没登记的按出现顺序排在后面，这样
        # 「有 finding、没 fact」的小节不会被挤到报告末尾。
        seen: List[str] = []
        for section, _, _ in self.facts:
            if section not in seen:
                seen.append(section)
        for f in self.findings:
            if f.section not in seen:
                seen.append(f.section)
        sections = [s for s in self.section_order if s in seen]
        sections += [s for s in seen if s not in sections]
        for section in sections:
            out.append(f"[{section}]")
            width = max(
                [_display_width(n) for s, n, _ in self.facts if s == section] or [0]
            )
            for s, name, value in self.facts:
                if s == section:
                    out.append(f"  {_pad(name, width)} {value}")
            for f in self.findings:
                if f.section != section:
                    continue
                out.append(f"  {_LEVEL_MARK[f.level]} {f.message}")
                for line in f.detail.splitlines():
                    if line.strip():
                        out.append(f"       {line}")
            out.append("")
        if self.n_errors:
            verdict = f"[ERR] {self.n_errors} 个错误"
            if self.n_warnings:
                verdict += f"、{self.n_warnings} 个警告"
        elif self.n_warnings:
            verdict = f"[WARN] {self.n_warnings} 个警告，没有错误"
        else:
            verdict = "[OK] 未发现问题"
        out.append(verdict)
        return "\n".join(out)

    def render_json(self) -> str:
        return json.dumps(
            {
                "title": self.title,
                "facts": [
                    {"section": s, "name": n, "value": v} for s, n, v in self.facts
                ],
                "findings": [f.as_dict() for f in self.findings],
                "n_errors": self.n_errors,
                "n_warnings": self.n_warnings,
            },
            ensure_ascii=False,
            indent=2,
        )

    def emit(self, as_json: bool) -> int:
        print(self.render_json() if as_json else self.render_text())
        return 1 if self.n_errors else 0


# ---------------------------------------------------------------------------
# 配置键的真源提取
# ---------------------------------------------------------------------------
def collect_consumed_config_keys() -> Dict[str, str]:
    """返回 {配置键: 来源说明}。

    静态提取 `runabfe.py`，不 import 就能跑（`doctor` 也不依赖它）：

    * `RunConfig.__init__` 里所有 `preset["<key>"] = ...` —— 这是命令行/配置
      文件真正落进运行配置的那批键；
    * `PRESET_CONFIGS` 字面量里的键 —— 预设自带的采样参数；
    * 模块里所有 `config.get("<key>")` / `config.data["<key>"]` —— 只读不写、
      但确实被消费的键；
    * `_EXTRA_CONSUMED_KEYS` —— 下游模块消费、本文件抓不到的那些。

    这么做的原因见模块 docstring 第 3 条：不在这里维护第二份参数表。
    """
    source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runabfe.py")
    keys: Dict[str, str] = {}
    try:
        tree = ast.parse(open(source_path, encoding="utf-8").read())
    except (OSError, SyntaxError):
        tree = None

    if tree is not None:
        for node in ast.walk(tree):
            # PRESET_CONFIGS = {"test": {...}, ...}
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "PRESET_CONFIGS"
                for t in node.targets
            ):
                if isinstance(node.value, ast.Dict):
                    for preset_body in node.value.values:
                        if isinstance(preset_body, ast.Dict):
                            for k in preset_body.keys:
                                if isinstance(k, ast.Constant) and isinstance(
                                    k.value, str
                                ):
                                    keys.setdefault(k.value, "预设 (PRESET_CONFIGS)")
            # preset["key"] = ... / config.data["key"] / config.get("key")
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                name = None
                if isinstance(node.value, ast.Name):
                    name = node.value.id
                elif isinstance(node.value, ast.Attribute):
                    name = node.value.attr
                if name in ("preset", "data") and isinstance(node.slice.value, str):
                    keys.setdefault(node.slice.value, "命令行/配置文件")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                base = node.func.value
                base_name = (
                    base.id
                    if isinstance(base, ast.Name)
                    else getattr(base, "attr", None)
                )
                if base_name == "config":
                    keys.setdefault(node.args[0].value, "运行期读取")

    for key in _EXTRA_CONSUMED_KEYS:
        keys.setdefault(key, "下游模块读取")
    return {k: v for k, v in keys.items() if not k.startswith("_")}


def collect_cli_schema() -> Dict[str, Dict[str, Any]]:
    """从 `runabfe.build_parser()` 这个**真** parser 上读出每个 dest 的约束。

    返回 {dest: {"type", "choices", "default", "help", "options"}}。
    子命令（prepare / refine-lambda-path / ...）的参数不在内——它们不是配置键。
    """
    runabfe = importlib.import_module("runabfe")
    parser = runabfe.build_parser()
    schema: Dict[str, Dict[str, Any]] = {}
    for action in parser._actions:  # argparse 没有公开 API，_actions 是稳定的
        if isinstance(action, argparse._SubParsersAction):
            continue
        if action.dest in ("help", argparse.SUPPRESS):
            continue
        schema[action.dest] = {
            "type": action.type,
            "choices": list(action.choices) if action.choices is not None else None,
            "default": action.default,
            "help": action.help or "",
            "options": list(action.option_strings),
        }
    return schema


# ---------------------------------------------------------------------------
# validate-config
# ---------------------------------------------------------------------------
def _check_value_against_schema(
    key: str, value: Any, entry: Dict[str, Any]
) -> Optional[Tuple[str, str]]:
    """返回 (message, detail) 或 None。只判 argparse 自己声明过的约束。"""
    choices = entry.get("choices")
    if choices is not None and value is not None and value not in choices:
        return (
            f"`{key}` 的值 {value!r} 不在允许取值里",
            f"允许: {', '.join(repr(c) for c in choices)}",
        )
    expected = entry.get("type")
    if expected in (int, float) and isinstance(value, bool):
        return (f"`{key}` 应为数值，实际是布尔 {value!r}", "")
    if expected is int and value is not None and not isinstance(value, int):
        return (f"`{key}` 应为整数，实际是 {type(value).__name__} {value!r}", "")
    if expected is float and value is not None and not isinstance(value, (int, float)):
        return (f"`{key}` 应为数值，实际是 {type(value).__name__} {value!r}", "")
    return None


def validate_config_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="runabfe.py validate-config",
        description="只读检查配置文件：未知键/拼写、类型与取值、输入路径、生效参数来源。"
        "不启动任何模拟，也不会改变运行期行为（拼写检查只在本命令里生效）。",
    )
    parser.add_argument("--config", default=None, help="要检查的 JSON/YAML 配置文件")
    parser.add_argument(
        "--preset",
        default="production",
        help="按哪个预设解释未在配置里出现的采样参数（默认 production）",
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args(argv)

    report = Report(
        "ABFE-IBS 配置检查 (validate-config)",
        section_order=(
            "配置文件",
            "未知键",
            "取值检查",
            "输入路径",
            "GROMACS",
            "生效参数",
            "随机种子",
        ),
    )
    runabfe = importlib.import_module("runabfe")

    # ---- 1. 载入 ----
    if not args.config:
        report.add(
            LEVEL_ERROR,
            "配置文件",
            "没有给 --config，无法检查",
            "用法: python runabfe.py validate-config --config abfe_config.json",
        )
        return report.emit(args.json)

    report.fact("配置文件", "路径", os.path.abspath(args.config))
    try:
        raw = runabfe._load_config(args.config)  # 与运行期同一个加载器
    except Exception as exc:
        report.add(
            LEVEL_ERROR,
            "配置文件",
            f"无法加载：{type(exc).__name__}",
            str(exc),
        )
        return report.emit(args.json)

    commented = sorted(k for k in raw if k.startswith("_"))
    active = {k: v for k, v in raw.items() if not k.startswith("_")}
    report.fact("配置文件", "生效键数", len(active))
    report.fact("配置文件", "注释键数", f"{len(commented)}（`_` 前缀，运行期跳过）")

    # argparse 的真 schema 先拿到：未知键那节要用它区分「拼错了」和「这是命令行选项、不是配置键」。
    try:
        schema = collect_cli_schema()
    except Exception as exc:  # parser 构造失败本身就是要报的问题
        schema = {}
        report.add(
            LEVEL_ERROR, "取值检查", f"无法构造 parser：{type(exc).__name__}", str(exc)
        )

    # ---- 2. 未知键与拼写 ----
    consumed = collect_consumed_config_keys()
    unknown = sorted(k for k in active if k not in consumed)
    if unknown:
        for key in unknown:
            cli_entry = schema.get(key)
            if cli_entry is not None:
                # 名字对得上一个命令行选项，但这个键不会从配置文件里被读走。
                # 这跟拼写错误不是一回事，别给"你是不是想写 xxx"的误导提示。
                options = " / ".join(cli_entry["options"]) or f"--{key}"
                report.add(
                    LEVEL_WARN,
                    "未知键",
                    f"`{key}` 只能从命令行给（{options}），配置文件里的这个键不会被读取",
                    f"当前值 {active[key]!r} 只会进配置快照；要生效请在命令行传 {options}。",
                )
                continue
            near = difflib.get_close_matches(key, list(consumed), n=3, cutoff=0.72)
            detail = (
                f"最接近的已知键: {', '.join(near)}"
                if near
                else "在已知配置键里找不到相近的名字"
            )
            report.add(
                LEVEL_ERROR if near else LEVEL_WARN,
                "未知键",
                f"`{key}` 不会被任何代码读取（值 {active[key]!r} 只会进配置快照）",
                detail,
            )
        report.add(
            LEVEL_WARN,
            "未知键",
            "运行期不会因为这些键拒绝启动——这是有意的",
            "启动期硬拒绝未知键曾经直接炸掉 resume（见 runabfe.py RunConfig 里 "
            "2026-08-24 的注释），所以拼写检查只放在本命令里。",
        )
    else:
        report.fact("未知键", "结果", "无")

    # ---- 3. 类型与取值 ----
    n_checked = 0
    for key, value in sorted(active.items()):
        entry = schema.get(key)
        if entry is None:
            continue
        n_checked += 1
        problem = _check_value_against_schema(key, value, entry)
        if problem:
            report.add(LEVEL_ERROR, "取值检查", problem[0], problem[1])
    report.fact("取值检查", "有 CLI 约束可比的键", n_checked)
    for key in ("temperature", "n_steps_per_window", "steps_per_update", "n_workers"):
        value = active.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0:
            report.add(LEVEL_ERROR, "取值检查", f"`{key}` 必须为正，实际 {value!r}")

    # OpenMM 平台名：`--platform` 在 argparse 里刻意没有 choices（不同 OpenMM
    # 构建能注册不同平台，写死会把合法值挡掉），所以这里只做常识提示，权威列表
    # 由 `doctor` 从 openmm 自己那里列出来。
    platform = active.get("platform")
    if isinstance(platform, str) and platform not in _WELL_KNOWN_OPENMM_PLATFORMS:
        report.add(
            LEVEL_WARN,
            "取值检查",
            f"`platform` = {platform!r} 不是常见的 OpenMM 平台名",
            f"常见值: {', '.join(_WELL_KNOWN_OPENMM_PLATFORMS)}；"
            "本机真实可用的平台用 `python runabfe.py doctor` 查。",
        )

    # ---- 4. 输入路径 ----
    n_paths = 0
    for key in _PATH_LIKE_INPUT_KEYS:
        value = active.get(key)
        if not isinstance(value, str) or not value:
            continue
        n_paths += 1
        if os.path.exists(value):
            report.fact("输入路径", key, value)
        else:
            report.add(
                LEVEL_ERROR,
                "输入路径",
                f"`{key}` 指向的路径不存在: {value}",
                "配置里写着但本机没有；换机器运行前必须核对。",
            )
    if not n_paths:
        report.fact("输入路径", "结果", "配置里没有输入路径字段")
    # 刚从 config-template 生成的文件这三个是 null（有意留空）。给 WARN 而不是
    # ERROR：`--resume` / `--analyze-only` 这两条路本来就不需要它们
    # （帮助文本写的是"首次运行时必需"）。
    unfilled = [k for k in ("gro", "top", "ligand") if not active.get(k)]
    if unfilled:
        report.add(
            LEVEL_WARN,
            "输入路径",
            f"这些首次运行必需的键还是空的: {', '.join(unfilled)}",
            "`--resume` / `--analyze-only` 不需要它们；要从 .gro/.top 建体系就必须填。",
        )

    # ---- 5. GROMACS include 目录 ----
    gmx_path = active.get("gmx_path")
    if gmx_path:
        if not os.path.exists(gmx_path):
            report.add(
                LEVEL_ERROR,
                "GROMACS",
                f"`gmx_path` 指向的路径不存在: {gmx_path}",
                "非 --openmm-cache-only 的路径会真的用到它；"
                "写 GROMACS 安装前缀或直接写 share/gromacs/top 都可以。",
            )
        else:
            resolved = runabfe.find_gmx_include_dir(gmx_path)
            report.fact("GROMACS", "gmx_path", gmx_path)
            report.fact("GROMACS", "解析到的 include 目录", resolved)
    else:
        resolved = runabfe.find_gmx_include_dir(None)
        report.fact("GROMACS", "gmx_path", "未设置")
        report.fact("GROMACS", "回退解析结果", resolved or "未找到")
        if resolved is None:
            report.add(
                LEVEL_WARN,
                "GROMACS",
                "没有 gmx_path，环境里也找不到力场 include 目录",
                "拓扑里若有需要它才能解析的 #include，运行会失败；"
                "用 --gmx-path 或 GMXLIB/GMXDATA 指定。",
            )

    # ---- 6. 生效参数与来源 ----
    presets = getattr(runabfe, "PRESET_CONFIGS", {})
    preset_body = presets.get(args.preset)
    if preset_body is None:
        report.add(
            LEVEL_ERROR,
            "生效参数",
            f"未知预设 {args.preset!r}",
            f"可选: {', '.join(sorted(presets))}",
        )
        preset_body = {}
    highlights = (
        "mode",
        "decoupling",
        "potential",
        "decharge_method",
        "platform",
        "temperature",
        "n_steps_per_window",
        "steps_per_update",
        "stage1_n_states",
        "stage2_n_states",
        "charge_treatment",
        "system_type",
        "repeat_seed",
    )
    for key in highlights:
        if key in active:
            source = "配置文件"
            value = active[key]
        elif key in preset_body:
            source = f"预设 {args.preset}"
            value = preset_body[key]
        elif key in schema:
            source = "argparse 默认值"
            value = schema[key]["default"]
        else:
            continue
        report.fact("生效参数", key, f"{value!r} ← {source}")
    report.add(
        LEVEL_WARN,
        "生效参数",
        "以上是「仅配置文件 + 预设」的解释结果，不含命令行覆盖",
        "命令行显式给的参数优先级更高（RunConfig 只在 flag 真的出现时才覆盖）。",
    )

    # ---- 7. seed 来源 ----
    env_seed = os.environ.get("ABFE_RANDOM_SEED")
    report.fact("随机种子", "ABFE_RANDOM_SEED", env_seed or "未设置")
    report.fact("随机种子", "config.repeat_seed", active.get("repeat_seed", "未设置"))

    return report.emit(args.json)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
# 版本契约：`tests/test_pymbar_version_and_uncertainty_contract.py` 把默认
# uncertainty_method 的等价性钉在这个版本上。装成别的版本不是"坏"，但
# 那条契约就不再被验证过，所以这里告警而不是沉默。
_PYMBAR_CONTRACT_VERSION = "4.2.0"

_DEPENDENCIES = (
    ("numpy", True),
    ("scipy", True),
    ("openmm", True),
    ("pymbar", True),
    ("mdtraj", False),
    ("torch", False),
    ("openmmml", False),
    ("yaml", False),
)


def _probe_module(name: str) -> Tuple[bool, str]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, str(getattr(module, "__version__", "版本未声明"))


def doctor_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="runabfe.py doctor",
        description="只读环境体检：解释器、依赖版本、OpenMM 平台、GPU、GROMACS、"
        "输出目录磁盘余量、原生插件。不建 Context、不启动模拟。",
    )
    parser.add_argument(
        "--output", default="./output", help="要检查磁盘余量的输出目录（默认 ./output）"
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args(argv)

    report = Report(
        "ABFE-IBS 环境体检 (doctor)",
        section_order=(
            "解释器",
            "依赖",
            "OpenMM",
            "GPU",
            "GROMACS",
            "磁盘",
            "原生插件",
        ),
    )

    # ---- 解释器 ----
    report.fact("解释器", "Python", sys.version.split()[0])
    report.fact("解释器", "可执行文件", sys.executable)
    if sys.version_info < (3, 10):
        report.add(
            LEVEL_ERROR,
            "解释器",
            f"Python {sys.version.split()[0]} 低于文档声明的 3.10+",
        )

    # ---- 依赖 ----
    for name, required in _DEPENDENCIES:
        ok, info = _probe_module(name)
        report.fact("依赖", name, info if ok else "未安装")
        if not ok and required:
            report.add(
                LEVEL_ERROR, "依赖", f"必需依赖 `{name}` 不可用", info
            )
        if ok and name == "pymbar" and info != _PYMBAR_CONTRACT_VERSION:
            report.add(
                LEVEL_WARN,
                "依赖",
                f"pymbar {info} 不是契约钉住的 {_PYMBAR_CONTRACT_VERSION}",
                "tests/test_pymbar_version_and_uncertainty_contract.py 的"
                "默认 uncertainty_method 等价性只在那个版本上验证过。",
            )

    # ---- OpenMM 平台 ----
    try:
        import openmm

        names = []
        for i in range(openmm.Platform.getNumPlatforms()):
            platform = openmm.Platform.getPlatform(i)
            names.append(f"{platform.getName()}(speed={platform.getSpeed():g})")
        report.fact("OpenMM", "版本", openmm.version.version)
        report.fact("OpenMM", "可用平台", ", ".join(names) or "无")
        available = {n.split("(")[0] for n in names}
        if "CUDA" not in available:
            report.add(
                LEVEL_WARN,
                "OpenMM",
                "没有 CUDA 平台；生产采样默认 --platform CUDA",
                "只做离线分析/测试可以忽略；要跑 MD 必须先修好 CUDA 平台。",
            )
    except Exception as exc:
        report.add(
            LEVEL_ERROR, "OpenMM", f"无法枚举平台：{type(exc).__name__}", str(exc)
        )

    # ---- GPU ----
    smi = shutil.which("nvidia-smi")
    if not smi:
        report.fact("GPU", "nvidia-smi", "不在 PATH 上")
    else:
        try:
            proc = subprocess.run(
                [
                    smi,
                    "--query-gpu=index,name,memory.used,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            for line in lines:
                report.fact("GPU", f"设备 {line.split(',')[0].strip()}", line)
            if not lines:
                report.add(LEVEL_WARN, "GPU", "nvidia-smi 没有列出任何设备")
        except Exception as exc:
            report.add(
                LEVEL_WARN, "GPU", f"nvidia-smi 调用失败：{type(exc).__name__}", str(exc)
            )

    # ---- GROMACS ----
    report.fact("GROMACS", "gmx", shutil.which("gmx") or "不在 PATH 上")
    report.fact("GROMACS", "GMXLIB", os.environ.get("GMXLIB") or "未设置")
    report.fact("GROMACS", "GMXDATA", os.environ.get("GMXDATA") or "未设置")
    try:
        runabfe = importlib.import_module("runabfe")
        resolved = runabfe.find_gmx_include_dir(None)
    except Exception as exc:
        resolved = None
        report.add(
            LEVEL_WARN,
            "GROMACS",
            f"include 目录解析失败：{type(exc).__name__}",
            str(exc),
        )
    report.fact("GROMACS", "无 gmx_path 时的解析结果", resolved or "未找到")
    if resolved is None:
        report.add(
            LEVEL_WARN,
            "GROMACS",
            "环境里找不到 GROMACS 力场 include 目录",
            "只有从 .gro/.top 建体系的路径需要它（--openmm-cache-only 和"
            "纯分析路径不需要）。要用就给 --gmx-path / config.gmx_path，"
            "或设置 GMXLIB / GMXDATA。",
        )

    # ---- 磁盘 ----
    probe_dir = args.output
    while probe_dir and not os.path.isdir(probe_dir):
        parent = os.path.dirname(os.path.abspath(probe_dir))
        if parent == probe_dir:
            break
        probe_dir = parent
    try:
        usage = shutil.disk_usage(probe_dir or ".")
        free_gb = usage.free / 2**30
        report.fact("磁盘", "探测目录", os.path.abspath(probe_dir or "."))
        report.fact("磁盘", "可用空间", f"{free_gb:.1f} GiB")
        if free_gb < 20:
            report.add(
                LEVEL_WARN,
                "磁盘",
                f"可用空间只剩 {free_gb:.1f} GiB",
                "一次两腿生产运行的轨迹/checkpoint 通常是几十 GiB 量级。",
            )
    except Exception as exc:
        report.add(
            LEVEL_WARN, "磁盘", f"无法读取磁盘用量：{type(exc).__name__}", str(exc)
        )

    # ---- 原生插件（residual sampling，默认不启用） ----
    plugin_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "plugins/LocalManyBodyResidual/build",
    )
    if os.path.isdir(plugin_dir):
        libs = sorted(f for f in os.listdir(plugin_dir) if f.endswith(".so"))
        report.fact("原生插件", "build 目录", plugin_dir)
        report.fact("原生插件", "已构建的库", ", ".join(libs) or "无 .so")
    else:
        report.fact("原生插件", "build 目录", "不存在（residual sampling 默认关闭，不影响生产路径）")

    return report.emit(args.json)

# ---------------------------------------------------------------------------
# config-template
# ---------------------------------------------------------------------------
# 模板里必须留空让用户自己填的键（体系相关，给默认值只会让人以为能直接跑）。
_REQUIRED_INPUT_KEYS = ("gro", "top", "ligand")

# 机器本地路径：绝不能从仓库配置里抄进模板（那份写的是
# `/home/ruigengji/gmx26.0C`，抄给别人就是错的）。
_MACHINE_LOCAL_KEYS = ("gmx_path",)

# 这些名字会被 `collect_consumed_config_keys()` 抓到，但**不是用户该写的配置键**，
# 所以 `config-template`（含 `--all`）不许输出它们。分两类：
#
# 1. **运行期写入的 provenance**——由代码 `config.data[...] = ...` 回填，用户
#    预先写进配置文件只会让快照说谎（例如 `repeat_seed_source` 声称 seed 来自
#    config，而实际 resolved seed 另有来源）。
# 2. **静态提取的误报**——`config.get("kwargs")` 里那个 `config` 是窗口级的
#    普通 dict（`ibs_engine.py:433`、`runabfe.py:4262`），不是 `RunConfig`；
#    `config.get("config")` 取的是 `--config` 自己的路径。提取器刻意宁宽勿窄
#    （宁可多认几个键，也不要把合法键误报成"未知键"），代价就是这里要挡一下。
#
# `validate-config` 不用这份名单：那边多认几个键只会少报一条噪音，不会出错。
_NON_USER_KEYS = frozenset(
    {
        "repeat_seed_source",
        "openmm_cache_only",
        "openmm_cache_only_complex_audit_sha256",
        "openmm_cache_only_solvent_audit_sha256",
        "outer_lambda_local_residual_ibs_identity",
        "config",
        "kwargs",
    }
)

# 模板的**键顺序与人工注释**取自这个文件——它是仓库里那份真实生产配置，
# 已经按输入/物理路线/采样预算/Boresch/... 分好组，而且 `_comment_*` 块是
# 人写的、比 argparse help 有信息量得多。
#
# 注意只取"键顺序 + 注释"，**不取值**：值一律用 preset/argparse 默认值重新
# 解析，否则会把 `gmx_path` 这种机器本地路径写进模板发给别人。
_TEMPLATE_LAYOUT_SOURCE = "abfe_config.json"

_TEMPLATE_HEADER = (
    "本文件由 `python runabfe.py config-template` 生成。键顺序与说明取自仓库的 "
    f"{_TEMPLATE_LAYOUT_SOURCE}；取值优先级为 预设 > 该文件的生产值 > argparse 默认，"
    "但 gro/top/ligand/gmx_path 一律留空需自行填写。"
    "`_` 前缀的键是注释，运行期会被跳过。填好 gro/top/ligand（以及需要时的 "
    "gmx_path）后用 `python runabfe.py validate-config --config <本文件>` 自查。"
    "命令行参数优先级高于本文件。"
)


def _resolve_template_value(
    key: str,
    schema: Dict[str, Dict[str, Any]],
    preset_body: Dict[str, Any],
    layout: Dict[str, Any],
) -> Any:
    """模板里该给这个键写什么值。

    取值优先级：**必填/机器本地键留空 > 预设 > 仓库生产配置 > argparse 默认**。

    为什么生产配置排在 argparse 默认之前：很多参数的 argparse 默认是 `None`，
    那是"命令行没给这个 flag"的哨兵，真正生效的默认在代码更深处
    （例如 `ibs_lse_log_residual_tolerance` 走 `defaults[...].default`）；
    还有的两边直接相反——`--enable-gradual-warmup` 是 `store_true`，argparse
    默认 `False`，而生产配置里是 `true`。照抄 argparse 默认会生成一份看着像
    "默认值"、实际跑不出生产行为的模板。仓库那份配置是实际跑过生产的值，
    所以它优先；只有机器本地路径不能抄（见 `_MACHINE_LOCAL_KEYS`）。
    """
    if key in _REQUIRED_INPUT_KEYS or key in _MACHINE_LOCAL_KEYS:
        return None
    if key in preset_body:
        return preset_body[key]
    if key in layout:
        return layout[key]
    if key in schema:
        return schema[key]["default"]
    return None


def config_template_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="runabfe.py config-template",
        description="打印一份配置模板：包含所有常用键、它们当前的默认值，以及仓库生产"
        "配置里那些人工写的说明。生成后用 validate-config 自查。",
    )
    parser.add_argument(
        "--preset",
        default="production",
        help="用哪个预设的采样参数填默认值（默认 production）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="把所有可消费的键都列出来（含不常用/实验开关），并附 argparse 帮助文本",
    )
    parser.add_argument("--out", default=None, help="写到文件而不是标准输出")
    args = parser.parse_args(argv)

    runabfe = importlib.import_module("runabfe")
    presets = getattr(runabfe, "PRESET_CONFIGS", {})
    if args.preset not in presets:
        print(
            f"[ERR] 未知预设 {args.preset!r}；可选: {', '.join(sorted(presets))}",
            file=sys.stderr,
        )
        return 1
    preset_body = presets[args.preset]
    consumed = collect_consumed_config_keys()
    try:
        schema = collect_cli_schema()
    except Exception as exc:
        print(f"[ERR] 无法构造 parser：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    layout_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), _TEMPLATE_LAYOUT_SOURCE
    )
    try:
        with open(layout_path, encoding="utf-8") as handle:
            layout = json.load(handle)  # dict 保序（py3.7+），顺序就是文件里的顺序
    except (OSError, ValueError) as exc:
        print(
            f"[ERR] 读不到布局来源 {_TEMPLATE_LAYOUT_SOURCE}：{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    template: Dict[str, Any] = {"_comment": _TEMPLATE_HEADER}
    emitted = set()
    pending_comments: Dict[str, Any] = {}
    for key, value in layout.items():
        if key == "_comment":
            # 布局来源自己的顶层说明不往下传：它指向 abfe_config.yaml，
            # 而仓库里没有这个文件。模板用上面 `_TEMPLATE_HEADER` 那段。
            continue
        if key.startswith("_"):
            pending_comments[key] = value
            continue
        if key not in consumed or key in _NON_USER_KEYS:
            # 例如 `preset`：它只是命令行选项名，写在配置文件里不会被读取
            # （validate-config 会专门指出这一点），模板不该教人写。
            pending_comments.clear()
            continue
        template.update(pending_comments)
        pending_comments.clear()
        template[key] = _resolve_template_value(key, schema, preset_body, layout)
        emitted.add(key)

    for key in _REQUIRED_INPUT_KEYS + _MACHINE_LOCAL_KEYS:
        if key in emitted:
            template[f"_comment_{key}"] = (
                "必填/机器相关：留空是有意的，请填成本机真实值"
                if key in _REQUIRED_INPUT_KEYS
                else "机器相关：填 GROMACS 安装前缀或 share/gromacs/top；"
                "也可改用 GMXLIB / GMXDATA 环境变量。用 doctor 查本机解析结果"
            )

    if args.all:
        extras = sorted(set(consumed) - emitted - _NON_USER_KEYS)
        if extras:
            template["_comment_extra_keys"] = (
                "以下是 --all 追加的键：代码会读它们，但不在仓库生产配置里出现，"
                "多为不常用或实验开关。不确定含义就删掉，让它走默认值。"
            )
            for key in extras:
                help_text = (schema.get(key) or {}).get("help") or ""
                if help_text:
                    template[f"_help_{key}"] = help_text
                template[key] = _resolve_template_value(key, schema, preset_body, layout)

    rendered = json.dumps(template, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(rendered)
        print(f"[OK] 已写出模板: {args.out}")
        print(f"   自查: python runabfe.py validate-config --config {args.out}")
    else:
        sys.stdout.write(rendered)
    return 0

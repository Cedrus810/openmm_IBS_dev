"""clone 下来必须 import 得动。

这道门防的是**发布时漏文件**：`git ls-files` 之外的本地模块被已跟踪代码 import，
本机跑得好好的（文件就在工作树里），clone 下来当场 `ModuleNotFoundError`。
实测过一次：`step_guard.py` 未跟踪，而 `ibs_engine.py:34` 是**模块级**
`from step_guard import guarded_step` ⟹ `import ibs_engine` / `abfe_pipeline` /
`abfe_preoptimizer` / `runabfe` 全部当场死。

分两层，因为后果不同：

* **模块级 import 未跟踪模块** = clone 当场死 ⟹ 硬失败。
* **函数内惰性 import 未跟踪模块** = import 过得去、跑到那一步才死 ⟹ 单独列出来，
  让发布的人自己决定那条路径要不要随发（例如默认关闭的可选功能）。

不在仓库里的第三方包不归这道门管（那是 `pyproject.toml` 的事）。
"""
import ast
import pathlib
import subprocess

import pytest

pytestmark = pytest.mark.cpu_only

REPO = pathlib.Path(__file__).resolve().parent.parent


def _tracked() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        pytest.skip("不是 git 工作区，或 git 不可用")
    return set(result.stdout.split())


def _local_target(name: str) -> str | None:
    """import 名 → 仓库内的本地模块/包路径；不是本地的返回 None。"""
    top = name.split(".")[0]
    if (REPO / f"{top}.py").is_file():
        return f"{top}.py"
    if (REPO / top).is_dir() and any((REPO / top).glob("*.py")):
        return f"{top}/"
    return None


def _is_tracked(target: str, tracked: set[str]) -> bool:
    if target.endswith("/"):
        return any(path.startswith(target) for path in tracked)
    return target in tracked


def _scan():
    tracked = _tracked()
    hard: list[str] = []
    lazy: list[str] = []
    for relative in sorted(p for p in tracked if p.endswith(".py")):
        path = REPO / relative
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        top_level = set(map(id, tree.body))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                target = _local_target(name)
                if target is None or _is_tracked(target, tracked):
                    continue
                entry = f"{relative}:{node.lineno}  →  {target}"
                (hard if id(node) in top_level else lazy).append(entry)
    return hard, lazy


def test_no_tracked_module_imports_an_untracked_module_at_import_time():
    """硬门：模块级 import 指向未跟踪模块 ⟹ clone 当场 ModuleNotFoundError。"""
    hard, _lazy = _scan()
    assert not hard, (
        "这些**模块级** import 指向 git 没跟踪的本地模块。clone 下来会当场 "
        "ModuleNotFoundError —— 发布前必须把它们和已跟踪文件同一批进去：\n  "
        + "\n  ".join(hard)
    )


def test_lazy_imports_of_untracked_modules_are_reported():
    """软门：惰性 import 只登记，不拦。

    跑到那一步才死，而且有些确实是"默认关闭、不随发"的可选功能。这条断言只在
    出现**新**的未登记项时才红 —— 登记表就是下面这份。
    """
    _hard, lazy = _scan()
    known_targets = {
        "lambda_path_versions.py",   # 主流水线会走到，发布必须带
        "multi_segment_analysis.py", # 同上
        "abfe_scripts/",                  # 重训链 ①②③④ + 自动重训的 manifest 生成
        "local_residual/",           # outer-λ 残差：只跟踪了 4/19
        "exp012_xed/",               # 仅 local_residual 的 ledger/metrics/mm_ledger/schema 需要
    }
    unknown = [
        entry for entry in lazy
        if entry.split("→")[-1].strip() not in known_targets
    ]
    assert not unknown, (
        "出现了新的「已跟踪代码惰性 import 未跟踪模块」。要么把它随发布带上，"
        "要么加进本测试的 known_targets 并写清为什么可以不带：\n  "
        + "\n  ".join(unknown)
    )

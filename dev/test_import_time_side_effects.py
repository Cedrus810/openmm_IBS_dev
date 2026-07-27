"""ATT-04 回归：import 期不得初始化 CUDA。

为什么这条重要：并行 stage worker 用 `mp.get_context("spawn")`
（`abfe_pipeline._run_stage_worker_process` / `ibs_engine`），spawn 反序列化
target 时必然 import `abfe_pipeline → ibs_engine → abfe_core`。只要 `abfe_core`
在模块级调 `torch.cuda.*`，每个子进程就都在 import 期抓一次 CUDA。

加重情节：子进程的 GPU 归属只通过 OpenMM 的 `props["DeviceIndex"]` 表达，
从不设 `CUDA_VISIBLE_DEVICES`——所以双 GPU 并行时**两个**子进程都会先在
device 0 上建 torch context，然后才各自去用被分配的那张 OpenMM 设备。

注意 TODO(ATT-04) 原本把这归咎于 `_run_stage_worker_process()` 里函数作用域的
`from abfe_pipeline import ABFEPipeline`。那是症状不是根因：spawn 无论如何都要
import 该模块才能取到 target，删掉那行不改变任何事。真正的根因是
`abfe_core` 的模块级 `GLOBAL_DEVICE, SUPPORTS_TF32 = get_optimal_device_settings()`。
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent


def test_importing_abfe_core_does_not_initialize_cuda():
    """在**干净子进程**里 import abfe_core，torch 的 CUDA context 必须仍未建立。

    必须开子进程：同一个 pytest 进程里别的测试可能已经碰过 CUDA，
    `torch.cuda.is_initialized()` 就不再是这次 import 的证据了。
    """
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "import importlib.util\n"
        "if importlib.util.find_spec('torch') is None:\n"
        "    print('SKIP:no-torch'); raise SystemExit(0)\n"
        "import torch\n"
        "assert not torch.cuda.is_initialized(), 'torch 自己就初始化了？'\n"
        "import abfe_core\n"
        "print('INITIALIZED' if torch.cuda.is_initialized() else 'CLEAN')\n"
    ) % str(REPO_ROOT)

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        pytest.skip(f"子进程 import 失败（缺依赖）: {proc.stderr.strip()[-400:]}")
    out = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if out == "SKIP:no-torch":
        pytest.skip("环境里没有 torch，这条断言无从谈起")
    assert out == "CLEAN", (
        "import abfe_core 之后 torch CUDA context 已建立——ATT-04 回归了。"
        "每个 spawn 子进程都会付这个代价，且双 GPU 时会一起挤到 device 0。"
    )


def _module_level_calls(path: Path):
    """返回模块顶层（不含函数/类体）里所有被调用的名字。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = []
    for node in tree.body:
        # 只看真正的顶层语句；def/class 内部是惰性的，不算 import 期副作用。
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                func = sub.func
                if isinstance(func, ast.Name):
                    names.append(func.id)
                elif isinstance(func, ast.Attribute):
                    names.append(func.attr)
    return names


def test_abfe_core_has_no_module_level_device_probe():
    """静态兜底：即使跑测试的机器没有 GPU，也要挡住这行被改回来。"""
    names = _module_level_calls(REPO_ROOT / "abfe_core.py")
    assert "get_optimal_device_settings" not in names, (
        "abfe_core.py 顶层又出现了 get_optimal_device_settings() 调用。"
        "它会在 import 期建 CUDA context；请用 get_global_device() / supports_tf32() "
        "惰性解析。"
    )
    for forbidden in ("is_available", "get_device_capability", "set_float32_matmul_precision"):
        assert forbidden not in names, (
            f"abfe_core.py 顶层出现了 torch.{forbidden}() 调用——import 期副作用"
        )


@pytest.mark.parametrize("module", ["ibs_engine.py", "abfe_pipeline.py", "abfe_core.py"])
def test_no_module_level_openmm_platform_access(module):
    """OpenMM 侧本来就是干净的（所有 Platform/Context 调用都在函数内），钉住它。"""
    names = _module_level_calls(REPO_ROOT / module)
    for forbidden in ("getPlatformByName", "getNumPlatforms", "Context"):
        assert forbidden not in names, (
            f"{module} 顶层出现了 OpenMM {forbidden}() 调用——spawn 子进程 import 期"
            "就会建 Context"
        )


def test_lazy_device_accessors_exist_and_are_memoized():
    import abfe_core

    assert hasattr(abfe_core, "get_global_device")
    assert hasattr(abfe_core, "supports_tf32")
    # 未调用前缓存必须是空的，否则说明还是 import 期就求值了。
    # （若本进程中已有别的测试调过，跳过而不是误报。）
    if abfe_core._DEVICE_SETTINGS_CACHE is None:
        assert abfe_core.get_global_device() in ("cpu", "cuda")
        first = abfe_core._DEVICE_SETTINGS_CACHE
        assert first is not None
        abfe_core.supports_tf32()
        assert abfe_core._DEVICE_SETTINGS_CACHE is first, "结果必须缓存，不能反复探测"

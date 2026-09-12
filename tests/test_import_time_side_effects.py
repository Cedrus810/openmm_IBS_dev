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
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    """惰性访问器必须存在，且**只探测一次**。

    2026-09-09 重写：原实现把整段断言包在
    `if abfe_core._DEVICE_SETTINGS_CACHE is None:` 里面。全量跑的时候别的测试早就
    把这个缓存填上了，于是这条测试只剩两句 `hasattr` —— 它名字里的 "memoized"
    从来没被验证过（顺序依赖的空测试）。现在显式把缓存清空再测，并在结束时恢复，
    这样它与测试执行顺序无关。

    "import 期不得求值"这一条由 `test_abfe_core_has_no_module_level_device_probe`
    （AST 扫模块顶层调用）负责，不再依赖"缓存此刻是不是空的"这种全局状态。
    """
    import abfe_core

    assert hasattr(abfe_core, "get_global_device")
    assert hasattr(abfe_core, "supports_tf32")

    saved = abfe_core._DEVICE_SETTINGS_CACHE
    try:
        abfe_core._DEVICE_SETTINGS_CACHE = None
        assert abfe_core.get_global_device() in ("cpu", "cuda")
        first = abfe_core._DEVICE_SETTINGS_CACHE
        assert first is not None, "调用过 get_global_device() 之后缓存仍为空"
        abfe_core.supports_tf32()
        assert abfe_core._DEVICE_SETTINGS_CACHE is first, "结果必须缓存，不能反复探测"
        # 同一个对象 ⟹ 第二个访问器没有重新探测设备。
        assert abfe_core.get_global_device() in ("cpu", "cuda")
        assert abfe_core._DEVICE_SETTINGS_CACHE is first
    finally:
        abfe_core._DEVICE_SETTINGS_CACHE = saved


# ---------------------------------------------------------------------------
# [P0-REMD-CUDA] pymbar 4 的后端是 JAX，JAX 默认预分配整卡 75% 显存
# ---------------------------------------------------------------------------


def test_jax_preallocation_is_disabled_before_pymbar_is_imported():
    """`XLA_PYTHON_CLIENT_PREALLOCATE=false` 必须在 import pymbar **之前**设好。

    实测（2026-08-04，`memtest/output_membrane_100ns`）：attachment 腿末尾用 pymbar
    解 BAR/MBAR，日志里 `JAX 64-bit mode is now on!` 之后紧跟着
        📊 [显存] Stage 0 attachment 结束: used=12197 free=3646 total=16303 MiB
    **12197 / 16303 = 74.8%**，正是 JAX 的默认 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`。
    于是 Stage 1 只剩 3646 MiB，而 12 个 replica Context 需要 12 × 317 = 3804 MiB，
    建满 11 个后第 12 个抛 `No compatible CUDA device is available`，
    整个 decharging 阶段静默退 CPU（慢约两个数量级）。

    环境变量必须在 JAX 被 import 之前生效——JAX 只在初始化时读一次。
    """
    source = (REPO_ROOT / "abfe_core.py").read_text(encoding="utf-8")

    setter = source.find('os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE"')
    assert setter >= 0, (
        "abfe_core.py 里不再设置 XLA_PYTHON_CLIENT_PREALLOCATE=false。"
        "去掉它 = JAX 会在解 MBAR 时预分配整卡 75% 显存，REMD 随后建不出 Context "
        "并静默退 CPU（P0-REMD-CUDA）。"
    )
    pymbar_import = source.find("import pymbar")
    assert pymbar_import >= 0
    assert setter < pymbar_import, (
        "XLA_PYTHON_CLIENT_PREALLOCATE 的设置跑到 import pymbar 之后了——"
        "JAX 只在初始化时读一次环境变量，设晚了等于没设。"
    )

    # 用 setdefault 而不是直接赋值：外部显式指定的值必须优先。
    assert 'os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] =' not in source, (
        "必须用 setdefault，让外部（例如显式导出 JAX_PLATFORMS=cpu 的人）能覆盖"
    )


def test_jax_backend_is_disabled_before_pymbar_is_imported():
    """`PYMBAR_DISABLE_JAX=1` 必须在 import pymbar **之前**设好。

    实测（2026-09-09，K=12 态 × N=9600 帧，生产那组 solver 参数）：

               wall     RSS       mmap 数    f_k
      JAX      6.30 s   887 MiB   1441       —
      numpy    0.37 s   120 MiB    891       max|Δf_k| = 8.9e-16

    JAX 后端慢 17 倍、胖 7 倍，答案逐位相同（每次解都要重新 XLA 编译，λ 表/
    帧数一变形状就变，编译缓存命不中；编译产物和 BFC 池从不归还）。留着它 =
    宿主内存被解算器啃光，然后由下游任何一次大分配替它抛 `std::bad_alloc`
    （2026-09-09 cyclod_ligand1/rep1：7.4 MB XML、93 GB 机器，崩在
    `XmlSerializer.deserializeSystem`）。
    """
    source = (REPO_ROOT / "abfe_core.py").read_text(encoding="utf-8")

    setter = source.find('os.environ.setdefault("PYMBAR_DISABLE_JAX"')
    assert setter >= 0, (
        "abfe_core.py 里不再设置 PYMBAR_DISABLE_JAX=1。去掉它 = 每次 MBAR 解都"
        "付一次 XLA 编译并永久留下几百 MiB，宿主内存耗尽后由下游随便哪次大分配"
        "抛无法归因的 std::bad_alloc（MBAR-JAX-HOST-MEM）。"
    )
    pymbar_import = source.find("import pymbar")
    assert pymbar_import >= 0
    assert setter < pymbar_import, (
        "PYMBAR_DISABLE_JAX 的设置跑到 import pymbar 之后了——pymbar 只在"
        "`mbar_solvers` 被 import 时读一次这个变量，设晚了等于没设。"
    )

    # 用 setdefault 而不是直接赋值：想要回 JAX 的人必须能用环境变量覆盖。
    assert 'os.environ["PYMBAR_DISABLE_JAX"] =' not in source, (
        "必须用 setdefault，让外部显式导出 PYMBAR_DISABLE_JAX=0 能要回 JAX"
    )


def test_pymbar_actually_runs_without_jax_after_importing_abfe_core():
    """光看源码不够：真的 import 进去，pymbar 必须落在 numpy 后端上。

    `mbar_solvers.force_no_jax` 是 pymbar 自己记录"我被要求不用 JAX"的标志；
    它为真时 `sys.modules` 里不该出现 `jax`（JAX 一旦初始化，那 18.4 GiB VSZ
    和几百 MiB RSS 就再也拿不回来了）。
    """
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "import os\n"
        "assert 'PYMBAR_DISABLE_JAX' not in os.environ, '外部已设，测不了默认行为'\n"
        "import abfe_core\n"
        "from pymbar import mbar_solvers\n"
        "print(mbar_solvers.force_no_jax, 'jax' in sys.modules)\n"
    ) % str(REPO_ROOT)

    env = dict(os.environ)
    env.pop("PYMBAR_DISABLE_JAX", None)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "True False", (
        f"pymbar 没有落在 numpy 后端上（force_no_jax, 'jax' in sys.modules）="
        f"{proc.stdout.strip().splitlines()[-1]!r}"
    )


def test_importing_abfe_core_sets_the_jax_preallocation_flag():
    """在干净子进程里 import abfe_core，标志必须已就位。"""
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "import os\n"
        "assert 'XLA_PYTHON_CLIENT_PREALLOCATE' not in os.environ, '外部已设，测不了默认行为'\n"
        "import abfe_core\n"
        "print(os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'))\n"
    ) % str(REPO_ROOT)

    env = dict(os.environ)
    env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(REPO_ROOT),
        env=env,
    )
    if proc.returncode != 0:
        pytest.skip(f"子进程 import 失败（缺依赖）: {proc.stderr.strip()[-400:]}")
    assert proc.stdout.strip().splitlines()[-1] == "false"

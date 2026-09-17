"""nvcc 发现不能只问 PATH；编译失败不得静默回退 CPU。

真机（计算节点，2026-09-15）：nvcc **只存在于 mamba env 里**，而作业用
**绝对路径调 env 的 python** 启动 —— 那样 env 的 `bin/` 不进 PATH。
同一份 launch.log 里 vanishing pilot 连起三次：

    第1次 properties={'Precision':'mixed','CudaCompiler':'nvcc'}  → CUDA
    第2次 properties={'Precision':'mixed','CudaCompiler':'nvcc'}  → CUDA
    第3次 default_program(325) 编译失败                          → 静默回退 CPU（87.6 s/λ）

⚠️⚠️ 更要紧的事实：**OpenMM 的 `CudaCompiler` 默认值是空串 = 默认走 NVRTC**，
nvcc 是可选旧路。所以「以前一直好好的」是碰巧命中 PATH 绕开了 NVRTC，
不是代码正确 —— 生成的 kernel 里那个 `0 }`（裸常数后面直接跟 `}`，缺分号）
是个真 bug，只是被 nvcc 盖住了。把 nvcc 找牢是**绕过**，不是修好。
"""
import os
import sys

import pytest

import abfe_pipeline as ap

pytestmark = pytest.mark.cpu_only


def test_nvcc_is_found_next_to_the_interpreter_without_path(monkeypatch, tmp_path):
    """conda/mamba env 里 nvcc 与 python 是邻居 —— 这条与有没有 activate 无关。"""
    envbin = tmp_path / "envs" / "x" / "bin"
    envbin.mkdir(parents=True)
    fake_py = envbin / "python"
    fake_py.write_text("")
    fake_nvcc = envbin / "nvcc"
    fake_nvcc.write_text("")
    fake_nvcc.chmod(0o755)

    monkeypatch.setattr(sys, "executable", str(fake_py))
    monkeypatch.setenv("PATH", "/nonexistent")      # PATH 里**没有** nvcc
    monkeypatch.delenv("OPENMM_CUDA_COMPILER", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    assert ap._resolve_nvcc() == str(fake_nvcc)


def test_an_explicit_env_var_wins(monkeypatch, tmp_path):
    exe = tmp_path / "my_nvcc"
    exe.write_text("")
    exe.chmod(0o755)
    monkeypatch.setenv("OPENMM_CUDA_COMPILER", str(exe))
    assert ap._resolve_nvcc() == str(exe)


def test_the_property_carries_an_absolute_path_not_a_bare_name(monkeypatch, tmp_path):
    """写绝对路径：`CudaCompiler` 最终由 system() 调起，不该再依赖子进程 PATH。"""
    exe = tmp_path / "nvcc"
    exe.write_text("")
    exe.chmod(0o755)
    monkeypatch.setenv("OPENMM_CUDA_COMPILER", str(exe))
    base, props = ap._build_platform_props("CUDA")
    assert base == "CUDA"
    assert props["CudaCompiler"] == str(exe)
    assert props["CudaCompiler"] != "nvcc"


def test_missing_nvcc_is_not_an_error_by_itself(monkeypatch):
    """找不到就不设属性（OpenMM 自己走 NVRTC）—— 不抛，也不假装找到了。"""
    monkeypatch.delenv("OPENMM_CUDA_COMPILER", raising=False)
    monkeypatch.setattr(ap, "_resolve_nvcc", lambda: None)
    base, props = ap._build_platform_props("CUDA")
    assert "CudaCompiler" not in props
    assert props["Precision"] == "mixed"


class _Boom(Exception):
    pass


def _ctx_factory_raising(msg):
    def _f(*a, **k):
        raise _Boom(msg)
    return _f


def test_a_compile_error_raises_instead_of_silently_using_cpu(monkeypatch):
    """编译错误是**确定性**的：回退 CPU = 一次静默的百倍减速，跑完还得重来。"""
    import openmm
    monkeypatch.setattr(openmm, "Context", _ctx_factory_raising(
        "Error compiling program: default_program(325): error: expected a \";\""))
    with pytest.raises(RuntimeError, match="编译失败"):
        ap._create_context_with_local_cpu_fallback(
            object(), lambda: object(), "CUDA", log=lambda *a: None)


def test_a_resource_error_still_falls_back_to_cpu(monkeypatch):
    """资源类（卡被占 / OOM）**保持**回退 —— 那类换台机就好。"""
    import openmm
    calls = {"n": 0}
    real_platform = openmm.Platform.getPlatformByName

    def _ctx(system, integrator, platform, props=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _Boom("CUDA error: out of memory")
        return "cpu-context"

    monkeypatch.setattr(openmm, "Context", _ctx)
    monkeypatch.setattr(openmm.Platform, "getPlatformByName",
                        staticmethod(lambda n: real_platform("CPU")))
    logged = []
    ctx, integ, base, props = ap._create_context_with_local_cpu_fallback(
        object(), lambda: object(), "CUDA", log=logged.append)
    assert ctx == "cpu-context" and base == "CPU" and props == {}
    assert any("资源类" in m for m in logged)
